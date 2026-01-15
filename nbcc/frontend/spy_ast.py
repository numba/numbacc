from __future__ import annotations
import ast as py_ast
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import spy.ast
from spy.analyze.symtable import Color, Symbol
from spy.fqn import FQN

if TYPE_CHECKING:
    from spy.vm.vm import SPyVM


@dataclass
class Node:
    """Simple node representation with opname and attributes dictionary."""

    _opname: str
    _attrdict: dict[str, Any]

    IGNORED = "loc"

    def __repr__(self):
        buf = [self._opname, "("]
        for k, v in self._attrdict.items():
            if k not in self.IGNORED:
                buf.append(f"{k}={v}, ")
        buf.append(")")
        return "".join(buf)

    def __hash__(self):
        return hash(id(self))

    def __getattr__(self, key: str) -> Any:
        try:
            return self._attrdict[key]
        except KeyError:
            raise AttributeError(key) from None

    def insert_fqn(self, fqn: FQN) -> Node:
        self._attrdict["fqn"] = fqn
        return self


def convert_to_node(
    node: Any,
    vm: "SPyVM",
    *,
    fields_to_ignore: Any = (),
) -> Node:
    dumper = Dumper(fields_to_ignore=fields_to_ignore, vm=vm)
    res = dumper.dump_anything(node)
    assert isinstance(res, Node)
    return res


class Dumper:
    """Convert AST nodes to simple Node representation"""

    fields_to_ignore: tuple[str, ...]
    vm: "SPyVM"

    def __init__(
        self,
        vm: "SPyVM",
        *,
        fields_to_ignore: Any = (),
    ) -> None:
        self.fields_to_ignore = tuple(fields_to_ignore)
        self.vm = vm

    def dump_anything(self, obj: Any) -> Node | list[Node]:
        if isinstance(obj, spy.ast.Node):
            return self.dump_spy_node(obj)
        elif isinstance(obj, py_ast.AST):
            return self.dump_py_node(obj)
        elif type(obj) is list:
            return self.dump_list(obj)
        elif type(obj) is Symbol:
            return self.dump_Symbol(obj)
        elif type(obj) in {str, type(None), int, float}:
            return obj
        else:
            return Node("literal", {"value": obj})

    def dump_spy_node(self, node: spy.ast.Node) -> Node:
        name = node.__class__.__name__
        fields = list(node.__class__.__dataclass_fields__)
        fields = [f for f in fields if f not in self.fields_to_ignore]
        return self._dump_node(node, name, fields)

    def dump_py_node(self, node: py_ast.AST) -> Node:
        name = "py:" + node.__class__.__name__
        fields = list(node.__class__._fields)
        fields = [f for f in fields if f not in self.fields_to_ignore]
        if isinstance(node, py_ast.Name):
            fields.append("is_var")
        return self._dump_node(node, name, fields)

    def dump_Symbol(self, sym: Symbol) -> Node:
        return Node(
            "Symbol",
            {
                "name": sym.name,
                "varkind": sym.varkind,
                "storage": sym.storage,
            },
        )

    def _dump_node(self, node: Any, name: str, fields: list[str]) -> Node:
        attrdict: dict[str, list[Node] | Node | str] = {}
        # Add color information from VM if available
        if self.vm and self.vm.ast_color_map:
            color: Optional[Color] = self.vm.ast_color_map.get(node, None)
            if color:
                attrdict["_color"] = color

        # Process each field
        for field in fields:
            value = getattr(node, field)
            attrdict[field] = self.dump_anything(value)

        return Node(name, attrdict)

    def dump_list(self, lst: list[Any]) -> list[Node]:
        items = [_assume_Node(self.dump_anything(item)) for item in lst]
        return items


def _assume_Node(x) -> Node:
    assert isinstance(x, Node)
    return x
