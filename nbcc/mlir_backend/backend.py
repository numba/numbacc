from __future__ import annotations

from typing import Sequence, cast

import mlir.dialects.arith as arith
import mlir.dialects.cf as cf
import mlir.dialects.func as func
import mlir.ir as ir
from mlir.dialects import llvm
from mlir.dialects.transform.interpreter import apply_named_sequence
from mlir.ir import _GlobalDebug
from sealir import ase
from sealir.dispatchtable import DispatchTableBuilder, dispatchtable
from spy.fqn import FQN

from nbcc.developer import TODO
from nbcc.mlir_utils import decode_type_name, decode_asm_operation
from nbcc.mlir_lowering import BackendInterface, MDMap, LowerStates

from ..frontend import grammar as sg, TranslationUnit
from .mlir_passes import PassManager

# ## MLIR Backend Implementation
#
# Define the core MLIR backend class that handles type lowering and
# expression compilation.


# _GlobalDebug.flag = True
_DEBUG = True


class Backend(BackendInterface):
    _tu: TranslationUnit
    _context: ir.Context

    Location = ir.Location
    InsertionPoint = ir.InsertionPoint

    def __init__(self, tu: TranslationUnit):
        self._tu = tu
        self._context = context = ir.Context()
        context.enable_multithreading(False)
        # context.allow_unregistered_dialects = True
        with context, ir.Location.name("Backend.__init__"):
            self.f32 = ir.F32Type.get(context=context)
            self._f64 = ir.F64Type.get(context=context)
            self.index_type = ir.IndexType.get(context=context)
            self.i8 = ir.IntegerType.get_signless(8, context=context)
            self._i32 = ir.IntegerType.get_signless(32, context=context)
            self._i64 = ir.IntegerType.get_signless(64, context=context)
            self._boolean = ir.IntegerType.get_signless(1, context=context)
            self._io_type = ir.IntegerType.get_signless(1, context=context)
            self._llvm_ptr = ir.Type.parse("!llvm.ptr")
            self._none_type = ir.Type.parse("!llvm.struct<()>")

    @classmethod
    def create(cls, tu: TranslationUnit) -> Backend:
        return cls(tu)

    def finalize_const_block(self, const_entry, target):
        # Use a break to jump from the constant block to the function block.
        # note that this is being inserted at end of constant block after the
        # Function construction when all the constants have been initialized.
        with const_entry:
            cf.br([], target)

    # Property accessors for BackendInterface compliance
    # Note: These properties expose existing instance attributes created in __init__
    # This satisfies the abstract properties defined in BackendInterface

    @property
    def context(self) -> ir.Context:
        return self._context

    @property
    def i32(self) -> ir.Type:
        return self._i32

    @property
    def i64(self) -> ir.Type:
        return self._i64

    @property
    def f64(self) -> ir.Type:
        return self._f64

    @property
    def boolean(self) -> ir.Type:
        return self._boolean

    @property
    def none_type(self) -> ir.Type:
        return self._none_type

    @property
    def io_type(self) -> ir.Type:
        return self._io_type

    @property
    def llvm_ptr(self) -> ir.Type:
        return self._llvm_ptr

    def initialize_io(self):
        return arith.constant(self.io_type, 0)

    def create_none(self):
        return llvm.mlir_zero(self.none_type)

    def create_return(self, values):
        return func.ReturnOp(values)

    def create_mlir_asm(self, opname, attr, result_types, args):
        assert isinstance(result_types, (list, tuple))
        if attr:
            irattrs = ir.Attribute.parse(attr)
            if isinstance(irattrs, ir.DictAttr):
                attrs = {
                    named_attr.name: named_attr.attr for named_attr in irattrs
                }
            else:
                raise ValueError("expects a dictattr")

        else:
            attrs = None
        print("DEBUG:", result_types)
        op = ir.Operation.create(opname, result_types, args, attributes=attrs)
        try:
            op.verify()
        except Exception:
            print(op.get_asm())
            raise
        if result_types:
            return op.result

    # Constant creation methods
    def create_constant_i32(self, value: int):
        return arith.constant(self.i32, value)

    def create_constant_i64(self, value: int):
        return arith.constant(self.i64, value)

    def create_constant_f64(self, value: float):
        return arith.constant(self.f64, value)

    def create_constant_boolean(self, value: bool):
        return arith.constant(self.boolean, value)

    # Control flow methods
    def create_if_op(self, condition, result_types, has_else=True):
        from mlir.dialects import scf

        return scf.IfOp(
            cond=condition, results_=result_types, hasElse=has_else
        )

    def create_yield_op(self, operands):
        from mlir.dialects import scf

        return scf.YieldOp(operands)

    def create_while_op(self, result_types, init_args):
        from mlir.dialects import scf

        return scf.WhileOp(results_=result_types, inits=init_args)

    def create_condition_op(self, condition, args):
        from mlir.dialects import scf

        return scf.ConditionOp(args=args, condition=condition)

    def get_scf_op_results(self, while_op):
        from mlir.dialects import scf

        return scf._get_op_results_or_values(while_op)

    # Function operation methods
    def create_function_call(self, result_types, callee, args):
        return func.call(result_types, callee, args)

    def create_function_declaration(
        self, name, arg_types, result_types, visibility="private"
    ):
        fntype = ir.FunctionType.get(arg_types, result_types)
        return func.FuncOp(name, fntype, visibility=visibility)

    # String constant method
    def create_string_constant(self, state: LowerStates, value: str):
        """Create string constant with LLVM global and address operations."""
        encoded = value.encode("utf8")
        length = len(encoded)

        struct_type = ir.Type.parse(
            f"!llvm.struct<(i64, array<{length} x i8>)>"
        )
        struct_value = ir.ArrayAttr.get(
            [
                ir.IntegerAttr.get(self.i64, length),
                ir.StringAttr.get(encoded),
            ]
        )

        # Use hash of string value for unique symbol name
        sym_name = ".const.str" + str(hash(value))

        # Create global in module body (current insertion point should be module body)
        llvm.GlobalOp(
            global_type=struct_type,
            sym_name=sym_name,
            linkage=ir.Attribute.parse("#llvm.linkage<private>"),
            constant=True,
            value=struct_value,
            addr_space=0,
        )
        with state.constant_block:
            # Return address operation
            ptr_type = self.llvm_ptr
            str_addr = llvm.AddressOfOp(
                ptr_type, ir.FlatSymbolRefAttr.get(sym_name)
            )
        return str_addr

    def create_function(self, function_name: str, input_types, output_types):
        # Constuct a function that emits a callable C-interface.
        fnty = func.FunctionType.get(input_types, output_types)
        fqn = FQN(function_name)
        if fqn.symbol_name == "main":
            TODO("XXX: hack main() function handling")
            export_name = "main"
        else:
            export_name = fqn.c_name
        TODO("TODO: is exporting logic")
        is_exporting = export_name == "main" or fqn.symbol_name.startswith(
            "export_"
        )
        fun = func.FuncOp(
            export_name,
            fnty,
            visibility=("public" if is_exporting else "private"),
        )
        if is_exporting:
            fun.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get()

        # Define two blocks within the function, a constant block to
        # define all the constants and a function block for the
        # actual content. This is done to prevent non-dominant use
        # of constants. (Use of a constant when declaration is done in
        # a region that isn't initialized.)
        const_block = fun.add_entry_block()
        fun.body.blocks.append(*[], arg_locs=None)
        func_block = fun.body.blocks[1]
        return fun, const_block, func_block

    def lower_type(self, ty: sg.TypeExpr) -> tuple[ir.Type, ...]:
        """Type Lowering

        Convert SealIR types to MLIR types for compilation.
        Always returns a tuple for consistent interface.
        """
        res = self._dispatch_lower_type(self, fqn=FQN(ty.name), args=ty.args)
        assert isinstance(res, tuple), res
        return res

    @dispatchtable
    @staticmethod
    def _dispatch_lower_type(disp: DispatchTableBuilder) -> None:
        @disp.default
        def _unknown_type(self, fqn: FQN, args: tuple):
            raise NotImplementedError(f"unknown type: {fqn}")

        def type_name_matches(fullname):
            def wrap(self, fqn: FQN, args: tuple) -> bool:
                return fqn.fullname == fullname

            return wrap

        def is_struct_type(self, fqn: FQN, args: tuple) -> bool:
            return self._tu.is_struct(fqn)

        @disp.case(type_name_matches("mlir::type::()"))
        def _handle_void(self, fqn: FQN, args: tuple):
            return ()

        @disp.case(
            lambda self, fqn, args: fqn.namespace.fullname == "mlir::type"
        )
        def _handle_mlir_types_by_parsing(self, fqn: FQN, args: tuple):
            [enc] = fqn.parts[-1].qualifiers
            tyname = decode_type_name(str(enc))
            ty = ir.Type.parse(tyname, context=self.context)
            return (ty,)

        def by_typename(fullname: str):
            def wrap(self, fqn, args):
                return fqn.fullname == fullname

            return wrap

        @disp.case(by_typename("builtins::i32"))
        def _handle_builtins_i32(self, fqn: FQN, args: tuple):
            return (self.i32,)

        @disp.case(by_typename("types::NoneType"))
        def _handle_none(self, fqn: FQN, args: tuple):
            return ()

        @disp.case(is_struct_type)
        def _handle_struct(self: Backend, fqn: FQN, args: tuple):
            struct = self._tu.get_struct(fqn)
            fields = list(struct.iterfields_w())
            is_lifted_type = len(fields) == 1 and fields[0].name == "__ll__"
            if is_lifted_type:
                [ll_field] = fields
                return self._dispatch_lower_type(
                    self, fqn=ll_field.w_T.fqn, args=()
                )
            else:
                TODO("regular non lifted struct type should go here")
                raise NotImplementedError("TODO")

    def handle_builtin_op(
        self, op_name: str, args, state: "LowerStates", lowering_instance=None
    ):
        return self._dispatch_handle_builtin_op(
            self, op_name, args, state, lowering_instance
        )

    @dispatchtable
    @staticmethod
    def _dispatch_handle_builtin_op(disp: DispatchTableBuilder) -> None:
        @disp.default
        def _unknown_builtin_op(
            self,
            op_name: str,
            args,
            state: "LowerStates",
            lowering_instance=None,
        ):
            raise NotImplementedError(f"Unhandled BuiltinOp {op_name!r}")

        def builtin_op_matches(op_name: str):
            def wrap(
                self,
                op_name_arg: str,
                args,
                state: "LowerStates",
                lowering_instance=None,
            ) -> bool:
                return op_name_arg == op_name

            return wrap

        @disp.case(builtin_op_matches("i32_add"))
        def _handle_i32_add(
            self,
            op_name: str,
            args,
            state: "LowerStates",
            lowering_instance=None,
        ):
            lhs, rhs = args
            return arith.addi(lhs, rhs)

        @disp.case(builtin_op_matches("i32_sub"))
        def _handle_i32_sub(
            self,
            op_name: str,
            args,
            state: "LowerStates",
            lowering_instance=None,
        ):
            lhs, rhs = args
            return arith.subi(lhs, rhs)

        @disp.case(builtin_op_matches("i32_lt"))
        def _handle_i32_lt(
            self,
            op_name: str,
            args,
            state: "LowerStates",
            lowering_instance=None,
        ):
            lhs, rhs = args
            return arith.cmpi(arith.CmpIPredicate.slt, lhs, rhs)

        @disp.case(builtin_op_matches("i32_gt"))
        def _handle_i32_gt(
            self,
            op_name: str,
            args,
            state: "LowerStates",
            lowering_instance=None,
        ):
            lhs, rhs = args
            return arith.cmpi(arith.CmpIPredicate.sgt, lhs, rhs)

        @disp.case(builtin_op_matches("i32_not"))
        def _handle_i32_not(
            self,
            op_name: str,
            args,
            state: "LowerStates",
            lowering_instance=None,
        ):
            (operand,) = args
            return arith.cmpi(
                arith.CmpIPredicate.eq,
                operand,
                arith.constant(self.i32, 0),
            )

        @disp.case(builtin_op_matches("print_i32"))
        def _handle_print_i32(
            self,
            op_name: str,
            args,
            state: "LowerStates",
            lowering_instance=None,
        ):
            io, operand = args

            print_fn = lowering_instance.declare_builtins(
                "spy_builtins$print_i32", [self.i32], []
            )
            func.call(
                print_fn.type.results, "spy_builtins$print_i32", [operand]
            )
            return io

        @disp.case(builtin_op_matches("print_str"))
        def _handle_print_str(
            self,
            op_name: str,
            args,
            state: "LowerStates",
            lowering_instance=None,
        ):
            io, operand = args

            print_fn = lowering_instance.declare_builtins(
                "spy_builtins$print_str", [self.llvm_ptr], []
            )

            func.call(
                print_fn.type.results, "spy_builtins$print_str", [operand]
            )
            return io

        @disp.case(builtin_op_matches("struct_make"))
        def _handle_struct_make(
            self,
            op_name: str,
            args,
            state: "LowerStates",
            lowering_instance=None,
        ):
            tys = [v.type for v in args]
            struct_type = llvm.StructType.get_literal(tys)

            struct_value = llvm.UndefOp(struct_type)
            for i, v in enumerate(args):
                struct_value = llvm.insertvalue(
                    struct_value, v, ir.DenseI64ArrayAttr.get([i])
                )
            return struct_value

        @disp.case(builtin_op_matches("struct_lift"))
        def _handle_struct_lift(
            self,
            op_name: str,
            args,
            state: "LowerStates",
            lowering_instance=None,
        ):
            [lifted] = [v for v in args]
            return lifted

        @disp.case(builtin_op_matches("struct_unlift"))
        def _handle_struct_unlift(
            self,
            op_name: str,
            args,
            state: "LowerStates",
            lowering_instance=None,
        ):
            [value] = args
            return value

        @disp.case(builtin_op_matches("struct_get"))
        def _handle_struct_get(
            self,
            op_name: str,
            args,
            state: "LowerStates",
            lowering_instance=None,
        ):
            struct_value, pos = args

            resty = self.i32  # HACK
            return llvm.extractvalue(
                resty, struct_value, ir.DenseI64ArrayAttr.get([pos])
            )

    def handle_mlir_op(self, mlir_op: str, resty, args):
        return self._dispatch_handle_mlir_op(self, mlir_op, resty, args)

    @dispatchtable
    @staticmethod
    def _dispatch_handle_mlir_op(disp: DispatchTableBuilder) -> None:
        @disp.default
        def _unknown_mlir_op(self, mlir_op: str, resty, args):
            raise NotImplementedError(f"Unhandled MLIR op {mlir_op!r}")

        def mlir_op_matches(op_name: str):
            def wrap(self, mlir_op: str, resty, args) -> bool:
                return mlir_op == op_name

            return wrap

        @disp.case(mlir_op_matches("tensor.add"))
        def _handle_tensor_add(self, mlir_op: str, resty, args):
            from mlir.dialects import arith, linalg, tensor

            [lhs, rhs] = args
            index = arith.constant(self.index_type, 0)
            dim = tensor.dim(args[0], index)
            out = tensor.empty([dim], element_type=self.f64)
            return linalg.add(lhs, rhs, outs=[out])

        @disp.case(mlir_op_matches("linalg.add"))
        def _handle_linalg_add(self, mlir_op: str, resty, args):
            from mlir.dialects import linalg

            # linalg.add needs a region
            [lhs, rhs, res] = args
            res = linalg.add(lhs, rhs, outs=[res])
            return res

        @disp.case(mlir_op_matches("linalg.sub"))
        def _handle_linalg_sub(self, mlir_op: str, resty, args):
            from mlir.dialects import linalg

            # linalg.sub needs a region
            [lhs, rhs, res] = args
            res = linalg.sub(lhs, rhs, outs=[res])
            return res

        @disp.case(mlir_op_matches("linalg.mul"))
        def _handle_linalg_mul(self, mlir_op: str, resty, args):
            from mlir.dialects import linalg

            # linalg.mul needs a region
            [lhs, rhs, res] = args
            res = linalg.mul(lhs, rhs, outs=[res])
            return res

        @disp.case(mlir_op_matches("linalg.div"))
        def _handle_linalg_div(self, mlir_op: str, resty, args):
            from mlir.dialects import linalg

            # linalg.div needs a region
            [lhs, rhs, res] = args
            res = linalg.div(lhs, rhs, outs=[res])
            return res

        @disp.case(mlir_op_matches("linalg.exp"))
        def _handle_linalg_exp(self, mlir_op: str, resty, args):
            from mlir.dialects import linalg

            # linalg.exp needs a region
            [src, res] = args
            res = linalg.exp(src, outs=[res])
            return res

        @disp.case(mlir_op_matches("mlir_linalg_reduce_sum_inner_keepdims"))
        def _handle_reduce_sum_inner_keepdims(self, mlir_op: str, resty, args):
            from mlir import ir
            from mlir.dialects import arith, linalg, tensor

            [arg] = args
            dtype = arg.type.element_type
            c0 = arith.constant(self.index_type, 0)
            dim = tensor.dim(arg, c0)
            init = tensor.empty(sizes=[dim], element_type=dtype)
            neg_inf = arith.constant(self.f64, float(0))
            init_filled = linalg.fill(neg_inf, outs=[init])
            max_reduce = linalg.reduce(
                result=[init.type],
                inputs=[arg],
                inits=[init_filled],
                dimensions=[1],
            )

            body = max_reduce.owner.regions[0].blocks.append(dtype, dtype)
            with self.InsertionPoint(body):
                linalg.YieldOp(
                    [arith.addf(body.arguments[0], body.arguments[1])]
                )

            assert max_reduce.owner.verify()

            c1 = arith.constant(self.index_type, 1)
            dim1 = tensor.dim(arg, c1)
            output = tensor.empty(sizes=(dim, dim1), element_type=dtype)
            assert output.owner.verify()

            bc = linalg.broadcast(
                input=max_reduce, outs=[output], dimensions=[1]
            )
            assert bc.verify()
            return bc

        @disp.case(mlir_op_matches("mlir_linalg_reduce_max_inner_keepdims"))
        def _handle_reduce_max_inner_keepdims(self, mlir_op: str, resty, args):
            from mlir import ir
            from mlir.dialects import arith, linalg, tensor

            [arg] = args
            dtype = arg.type.element_type
            c0 = arith.constant(self.index_type, 0)
            dim = tensor.dim(arg, c0)
            init = tensor.empty(sizes=[dim], element_type=dtype)
            neg_inf = arith.constant(self.f64, float("-inf"))
            init_filled = linalg.fill(neg_inf, outs=[init])
            max_reduce = linalg.reduce(
                result=[init.type],
                inputs=[arg],
                inits=[init_filled],
                dimensions=[1],
            )

            body = max_reduce.owner.regions[0].blocks.append(dtype, dtype)
            with self.InsertionPoint(body):
                linalg.YieldOp(
                    [arith.maximumf(body.arguments[0], body.arguments[1])]
                )

            assert max_reduce.owner.verify()

            c1 = arith.constant(self.index_type, 1)
            dim1 = tensor.dim(arg, c1)
            output = tensor.empty(sizes=(dim, dim1), element_type=dtype)
            assert output.owner.verify()

            bc = linalg.broadcast(
                input=max_reduce, outs=[output], dimensions=[1]
            )
            assert bc.verify()
            return bc

    def get_ll_type(self, expr: ase.SExpr, mdmap: MDMap) -> ir.Type:
        mds = mdmap.lookup_typeinfo(expr)
        if not mds:
            return None
        [ty] = mds
        [llty] = self.lower_type(cast(sg.TypeExpr, ty.type_expr))
        return llty

    def make_module(self, module_name: str) -> ir.Module:
        with self.context:
            return ir.Module.create(loc=ir.Location.name(module_name))

    def _make_pass_pipeline(self, *passes, with_subprocess=True):
        return PassManager(passes, with_subprocess=with_subprocess)

    def run_passes(
        self, module: ir.Module, transforms: dict[str, Sequence[str]]
    ) -> ir.Module:
        """MLIR Pass Pipeline

        Apply MLIR passes for optimization and lowering to LLVM IR.
        """
        from . import mlir_passes as mp

        for name in transforms:
            self._add_noinline_to_callsite(module, name)

        module = self._make_pass_pipeline(
            mp.Canonicalize(),
            mp.Inline(),
        ).run(module.operation)

        print("After Phase 1")

        for fname, pass_seq in transforms.items():
            self._run_per_function_transform(module, fname, pass_seq)

        module = self._make_pass_pipeline(
            mp.Canonicalize(),
            mp.LinalgGeneralizeNamedOps(),
            mp.LinalgFuseElementwiseOps(),
            mp.LoopInvariantCodeMotion(),
            mp.FoldMemRefAliasOps(),
            # Vector
            mp.LowerVectorMask(),
            mp.Canonicalize(),
            mp.FoldTensorSubsetOps(),  # folds tensor-slice into vector-transfer
            mp.Canonicalize(),
        ).run(module.operation)
        print("After Phase 3 (cleanup)")

        module = self._make_pass_pipeline(
            mp.EliminateEmptyTensors(),
            mp.EmptyTensorToAllocTensor(),
            # Bufferization
            mp.OneShotBufferize(bufferize_function_boundaries=True),
            mp.ConvertVectorToSCF(),
            mp.Canonicalize(),
            mp.CSE(),
        ).run(module.operation)

        print("After Phase 4 (bufferize)")

        module = self._make_pass_pipeline(
            # Affine passes goes after Bufferize
            mp.ConvertLinalgToAffineLoops(),
            mp.NormalizeMemRefs(),
            mp.Canonicalize(),
            mp.CSE(),
            mp.SymbolDCE(),
            mp.ExpandStridedMetadata(),
            mp.AffineScalrep(),
            mp.AffineSimplifyStructures(),
            mp.AffineLoopFusion(mode="greedy", maximal=1),
            # mp.ConvertLinalgToParallelLoops(),
            mp.LowerAffine(),
            mp.Canonicalize(),
            mp.PromoteBuffersToStack(),
            mp.BufferHoisting(),
            mp.BufferLoopHoisting(),
            mp.FoldMemRefAliasOps(),
            # The CSE and canonicalize take care of the reminding redundant memref ops
            mp.CSE(),
            mp.Canonicalize(),
        ).run(module.operation)

        print("After Phase 4.1 (SCF ops)")

        module = self._make_pass_pipeline(
            mp.ScfForLoopCanonicalization(),
            mp.ScfForLoopRangeFolding(),
            mp.ScfForLoopToParallel(),
            mp.ScfParallelLoopFusion(),
            mp.Canonicalize(),
        ).run(module.operation)

        print("After Phase 5 (prelower)")

        module = self._make_pass_pipeline(
            mp.OwnershipBasedBufferDeallocation(),
            mp.BufferDeallocationSimplification(),
            mp.BufferizationLowerDeallocations(),
            mp.ConvertBufferizationToMemRef(),
            # Lowering
            mp.Canonicalize(),
            mp.ConvertSCFToCF(),
            mp.ConvertVectorToLLVM(enable_arm_neon=True),
            mp.FinalizeMemRefToLLVM(),
            mp.ConvertMathToLibM(),
            mp.ConvertFuncToLLVM(),
            mp.ConvertIndexToLLVM(),
            mp.ConvertArithToLLVM(),
            mp.ConvertCFToLLVM(),
            mp.ReconileUnrealizedCasts(),
        ).run(module.operation)
        return module

    def _add_noinline_to_callsite(self, module: ir.Module, fname: str):
        def iterate_funcop(module: ir.Module):
            for blk in module.body.region.blocks:
                for modop in blk.operations:
                    if isinstance(modop, func.FuncOp):
                        yield modop

        for f_op in iterate_funcop(module):
            if f_op.name.value == fname:
                attrmap = f_op.attributes
                attrmap["no_inline"] = ir.UnitAttr.get(context=module.context)

    def _run_per_function_transform(
        self, module: ir.Module, fname: str, pass_seq: Sequence[str]
    ):

        def load_transform(tmod: ir.Module):
            [transform_op] = tmod.body.operations

            def runner(payload):
                return apply_named_sequence(payload, transform_op, tmod)

            return runner

        from . import transforms

        for pass_name in pass_seq:
            tmod = ir.Module.parse(
                getattr(transforms, pass_name), context=self.context
            )
            transform = load_transform(tmod)

            for op in module.body.region.blocks[0].operations:
                if op.name.value == fname:
                    fn_op = op
                    break

            transform(fn_op.operation)

            print(f"Transformed: {fname} after {pass_name}")
            print(fn_op.operation.get_asm())
