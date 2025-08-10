import ast
import dataclasses

import pytest

import gstaichi as ti
from gstaichi.lang.ast.ast_transformers.call_transformer import CallTransformer
from tests import test_utils


@dataclasses.dataclass
class MyStructAB:
    a: ti.types.NDArray[ti.i32, 1]
    b: ti.types.NDArray[ti.i32, 1]


@dataclasses.dataclass
class MyStructCD:
    c: ti.types.NDArray[ti.i32, 1]
    d: ti.types.NDArray[ti.i32, 1]
    my_struct_ab: MyStructAB


@dataclasses.dataclass
class MyStructEF:
    e: ti.types.NDArray[ti.i32, 1]
    f: ti.types.NDArray[ti.i32, 1]
    my_struct_cd: MyStructCD


def dump_ast_list(nodes: list[ast.AST]) -> str:
    res_l = []
    for node in nodes:
        res_l.append(ast.dump(node, indent=2))
    return "[\n" + ",\n".join(res_l) + "\n]"


@pytest.mark.parametrize(
    "args_in, expected_args_out",
    [
        (
            [ast.Name(id="my_struct_ab", ctx=ast.Load(), ptr=MyStructAB)],
            [
                ast.Name(id="__ti_my_struct_ab__ti_a", ctx=ast.Load()),
                ast.Name(id="__ti_my_struct_ab__ti_b", ctx=ast.Load()),
            ],
        ),
        (
            [ast.Name(id="my_struct_cd", ctx=ast.Load(), ptr=MyStructCD)],
            [
                ast.Name(id="__ti_my_struct_cd__ti_c", ctx=ast.Load()),
                ast.Name(id="__ti_my_struct_cd__ti_d", ctx=ast.Load()),
                ast.Name(id="__ti_my_struct_cd__ti_my_struct_ab__ti_a", ctx=ast.Load()),
                ast.Name(id="__ti_my_struct_cd__ti_my_struct_ab__ti_b", ctx=ast.Load()),
            ],
        ),
        (
            [ast.Name(id="my_struct_ef", ctx=ast.Load(), ptr=MyStructEF)],
            [
                ast.Name(id="__ti_my_struct_ef__ti_e", ctx=ast.Load()),
                ast.Name(id="__ti_my_struct_ef__ti_f", ctx=ast.Load()),
                ast.Name(id="__ti_my_struct_ef__ti_my_struct_cd__ti_c", ctx=ast.Load()),
                ast.Name(id="__ti_my_struct_ef__ti_my_struct_cd__ti_d", ctx=ast.Load()),
                ast.Name(id="__ti_my_struct_ef__ti_my_struct_cd__ti_my_struct_ab__ti_a", ctx=ast.Load()),
                ast.Name(id="__ti_my_struct_ef__ti_my_struct_cd__ti_my_struct_ab__ti_b", ctx=ast.Load()),
            ],
        ),
    ],
)
@test_utils.test()
def test_expand_Call_dataclass_args(args_in: list[ast.AST], expected_args_out: list[ast.AST]) -> None:
    for arg in args_in:
        arg.lineno = 1
        arg.endlineno = 2
        arg.col_offset = 1
        arg.end_col_offset = 2
    args_out = CallTransformer._expand_Call_dataclass_args(args_in)
    print("args_in", dump_ast_list(args_in))
    print("args_out", dump_ast_list(args_out))
    print("expected_args_out", dump_ast_list(expected_args_out))
    assert dump_ast_list(expected_args_out) == dump_ast_list(args_out)
