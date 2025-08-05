import gstaichi as ti


def test_binding():
    ti.init()
    gstaichi_lang = ti._lib.core
    print(gstaichi_lang.BinaryOpType.mul)
    one = gstaichi_lang.make_const_expr_int(ti.i32, 1)
    two = gstaichi_lang.make_const_expr_int(ti.i32, 2)
    expr = gstaichi_lang.make_binary_op_expr(gstaichi_lang.BinaryOpType.add, one, two)
    print(gstaichi_lang.make_global_store_stmt(None, None))
