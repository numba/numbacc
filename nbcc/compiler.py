import logging
import os
import subprocess as subp
import sys
import tempfile
from contextlib import ExitStack
from pathlib import Path
from pprint import pprint
from typing import cast, Sequence

import sealir.rvsdg.grammar as rg
import spy
from egglog import EGraph
from mlir import ir
from sealir.ase import SExpr, TapeCrawler
from sealir.eqsat.rvsdg_convert import egraph_conversion
from sealir.eqsat.rvsdg_eqsat import GraphRoot
from sealir.eqsat.rvsdg_extract import CostModel as _CostModel
from sealir.eqsat.rvsdg_extract import egraph_extraction
from sealir.rvsdg import format_rvsdg

from nbcc.developer import TODO
from nbcc.egraph.conversion import ExtendEGraphToRVSDG
from nbcc.egraph.rules import egraph_convert_metadata, egraph_optimize
from nbcc.frontend import TranslationUnit, frontend
from nbcc.frontend.grammar import IRTag, TypeInfo
from nbcc.mlir_backend.backend import Backend, Lowering, MDMap

logging.disable(logging.INFO)


def compile(path: str, out_path: str) -> None:
    module = compile_to_mlir(path)
    make_binary(module, out_path)


def compile_shared_lib(path: str, out_path: str) -> None:
    module = compile_to_mlir(path)
    make_shared(module, out_path)


def compile_to_mlir(path: str) -> ir.Module:
    tu = frontend(path)

    func_map: dict[str, rg.Func]
    func_map, mdlist = middle_end(tu)
    pprint(func_map)
    be = Backend()
    mdmap = MDMap()
    mdmap.load(mdlist)

    module = be.make_module(path)

    transform_map: dict[str, Sequence[str]] = {}
    for fname, rvsdg_ir in func_map.items():
        lowering = Lowering(be, module, mdmap, func_map)
        TODO("Not handling lowering argtypes")
        fn_op = lowering.lower(rvsdg_ir)
        print(fn_op.operation.get_asm())

        irtags = lowering.irtags(rvsdg_ir)
        print("== IRTAGS", irtags)
        if mlir_transforms := irtags.get("mlir.transforms"):
            transform_map[fn_op.name.value] = [v for k, v in mlir_transforms]

    lowering.module.operation.verify()

    print("-------------")
    lowering.module.dump()

    print("=============")
    print(lowering.module.operation.get_asm())
    pprint(transform_map)
    module = be.run_passes(module, transforms=transform_map)
    print("After optimization")
    print(module)

    return module


def make_binary(module: ir.Module, out_path: str):
    with ExitStack() as raii:
        temp_file_mlir = raii.enter_context(
            tempfile.NamedTemporaryFile(suffix=".mlir", mode="w")
        )
        print(
            module.operation.get_asm(enable_debug_info=True),
            file=temp_file_mlir,
        )
        temp_file_mlir.flush()

        temp_file_llvmir = raii.enter_context(
            tempfile.NamedTemporaryFile(suffix=".ll", mode="w")
        )
        subp.check_call(
            [
                "mlir-translate",
                "--mlir-to-llvmir",
                temp_file_mlir.name,
                "-o",
                temp_file_llvmir.name,
            ]
        )
        # subp.check_call(["cat", "out.ll"])
        subp.check_call(
            [
                "clang",
                "-o",
                out_path,
                temp_file_llvmir.name,
                "-Ldeps/spy/spy/libspy/build/native/release/",
                "-lspy",
            ]
        )


def make_shared(module: ir.Module, out_path: str):
    from ctypes.util import find_library

    lib_path = find_library("mlir_c_runner_utils")
    if lib_path is None:
        raise RuntimeError("Could not find mlir_c_runner_utils library")
    libdir = os.path.dirname(lib_path)
    with ExitStack() as raii:
        temp_file_mlir = raii.enter_context(
            tempfile.NamedTemporaryFile(suffix=".mlir", mode="w")
        )
        print(
            module.operation.get_asm(enable_debug_info=True),
            file=temp_file_mlir,
        )
        temp_file_mlir.flush()

        temp_file_llvmir = raii.enter_context(
            tempfile.NamedTemporaryFile(suffix=".orig.ll", mode="w")
        )
        temp_file_llvm_opt = raii.enter_context(
            tempfile.NamedTemporaryFile(suffix=".opt.ll", mode="w")
        )
        temp_file_native_obj = raii.enter_context(
            tempfile.NamedTemporaryFile(suffix=".o", mode="wb")
        )
        target_triple = subp.check_output(
            "llvm-config --host-target".split(), encoding="utf8"
        ).strip()
        subp.check_call(
            [
                "mlir-translate",
                "--mlir-to-llvmir",
                temp_file_mlir.name,
                "-o",
                temp_file_llvmir.name,
            ]
        )

        print(temp_file_llvmir.name.center(80, "-"))
        subp.check_call(["cat", temp_file_llvmir.name])

        subp.check_call(
            [
                "opt",
                "-passes=default<O3>",
                "-S",
                f"-mtriple={target_triple}",
                temp_file_llvmir.name,
                "-o",
                temp_file_llvm_opt.name,
            ]
        )

        print(temp_file_llvm_opt.name.center(80, "-"))
        subp.check_call(["cat", temp_file_llvm_opt.name])

        print(80 * "=")
        subp.check_call(
            [
                "llc",
                "-O3",
                "-filetype=obj",
                "--relocation-model=pic",
                temp_file_llvm_opt.name,
                "-o",
                temp_file_native_obj.name,
            ]
        )
        spydir = os.path.dirname(spy.__file__)
        spylinkdir = Path(spydir) / "libspy" / "build" / "native" / "release"
        subp.check_call(
            [
                "clang",
                "-shared",
                "-o",
                out_path,
                temp_file_native_obj.name,
                f"-L{spylinkdir}",
                "-lspy",
                f"-L{libdir}",
                f"-lmlir_c_runner_utils",
            ]
        )


def middle_end(
    tu: TranslationUnit,
) -> tuple[dict[str, rg.Func], list[TypeInfo | IRTag]]:
    func_nodes: dict[str, rg.Func] = {}
    mdlist: list[TypeInfo | IRTag] = []

    for fqn in tu.list_functions():
        fi = tu.get_function(fqn)
        print(fi.fqn, fi.region)

        memo = egraph_conversion(fi.region)

        root = GraphRoot(memo[fi.region])

        egraph = EGraph()
        egraph.let("root", root)
        egraph.let("mds", egraph_convert_metadata(fi.metadata, memo))

        expand_struct_type(tu, egraph)

        egraph_optimize(egraph)

        extraction = egraph_extraction(egraph, cost_model=CostModel())
        extraction.compute()
        extresult = extraction.extract_common_root()
        print("egraph extracted")
        print("cost", extresult.cost)

        tape = fi.region._tape
        last = tape.last
        converted_root: SExpr = extresult.convert(
            fi.region, ExtendEGraphToRVSDG
        )

        for node in converted_root._args:
            match node:
                case rg.Func(fname=str(matched_fqn)):
                    assert matched_fqn not in func_nodes
                    func_nodes[matched_fqn] = node

        crawler = TapeCrawler(tape, converted_root._get_downcast())
        crawler.move_to_pos_of(last)
        crawler.move_to_first_record()

        for rec in crawler.walk():
            node = rec.to_expr()
            if isinstance(node, TypeInfo):
                mdlist.append(node)
            elif isinstance(node, IRTag):
                mdlist.append(node)

    assert len(func_nodes) >= 1
    for func in func_nodes.values():
        print(format_rvsdg(cast(SExpr, func)))

        print(cast(SExpr, func)._tape.dump())
    return func_nodes, mdlist


class CostModel(_CostModel):
    def get_cost_function(
        self,
        nodename,
        op,
        ty,
        cost,
        children,
    ):
        if op in ["Py_Call", "Py_LoadGlobal"]:
            return self.get_simple(10000)
        elif op in ["CallFQN"]:
            return self.get_simple(1)
        else:
            return super().get_cost_function(nodename, op, ty, cost, children)


def expand_struct_type(tu: TranslationUnit, egraph):
    from egglog import Ruleset

    from nbcc.egraph.rules import (
        create_ruleset_struct__get_field__,
        create_ruleset_struct__make__,
        create_ruleset_struct__lift__,
        create_ruleset_struct__unlift__,
    )

    schedule = Ruleset(None)  # empty ruleset
    for fqn_struct, w_obj_struct in tu._structs.items():
        print(fqn_struct, w_obj_struct)

        is_lifted_type = "__ll__" in w_obj_struct.dict_w
        for fqn, w_obj in tu._builtins.items():
            if fqn_struct.fullname.startswith(fqn_struct.fullname):
                print("BUITIN", fqn)
                subname = fqn.parts[-1].name
                if subname == "__make__":
                    if is_lifted_type:
                        print("Add __lift__")
                        schedule |= create_ruleset_struct__lift__(w_obj)

                    else:
                        print("Add __make__")
                        schedule |= create_ruleset_struct__make__(w_obj)

                elif subname.startswith("__get_"):

                    if is_lifted_type:
                        assert subname == "__get___ll____"
                        schedule |= create_ruleset_struct__unlift__(w_obj)
                    else:
                        print("Add field getter")
                        for i, w_field in enumerate(
                            w_obj_struct.iterfields_w()
                        ):
                            if subname == f"__get_{w_field.name}__":
                                schedule |= create_ruleset_struct__get_field__(
                                    w_obj, i
                                )

    egraph.run(schedule.saturate())


if __name__ == "__main__":
    argv = sys.argv[1:]
    if "-shared" in argv:
        argv.remove("-shared")
        compile_shared_lib(*argv)
    else:
        compile(*argv)
