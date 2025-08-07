# TODO: rename this to test_py_dataclasses (not doing now because would make PR review harder)
import gc
from dataclasses import dataclass
from typing import Any

import pytest

import gstaichi as ti

from tests import test_utils


@pytest.fixture
def ti_type(use_ndarray: bool) -> Any:
    if use_ndarray:
        return ti.ndarray
    return ti.field


@pytest.fixture
def ti_annotation(use_ndarray: bool) -> Any:
    class TiTemplateBuilder:
        """
        Allows ti_annotation[ti.i32, 2] to be legal
        """

        def __getitem__(self, _):
            return ti.Template

    if use_ndarray:
        return ti.types.ndarray
    return TiTemplateBuilder()


@test_utils.test()
def test_ndarray_struct_kwargs():
    gc.collect()
    gc.collect()

    a = ti.ndarray(ti.i32, shape=(55,))
    b = ti.ndarray(ti.i32, shape=(57,))
    c = ti.ndarray(ti.i32, shape=(211,))
    d = ti.ndarray(ti.i32, shape=(223,))
    e = ti.ndarray(ti.i32, shape=(227,))

    @dataclass
    class MyStruct:
        a: ti.types.NDArray[ti.i32, 1]
        b: ti.types.NDArray[ti.i32, 1]
        c: ti.types.NDArray[ti.i32, 1]

    @ti.func
    def s4(a: ti.types.NDArray[ti.i32, 1], b: ti.types.NDArray[ti.i32, 1]) -> None:
        a[1] += 888
        b[2] += 999

    @ti.func
    def s3(z3: ti.types.NDArray[ti.i32, 1], my_struct3: MyStruct, bar3: ti.types.NDArray[ti.i32, 1]) -> None:
        z3[25] += 90
        my_struct3.a[47] += 42
        my_struct3.b[49] += 43
        my_struct3.c[43] += 44
        bar3[113] += 125
        s4(my_struct3.a, my_struct3.b)

    @ti.func
    def s2(z3: ti.types.NDArray[ti.i32, 1], my_struct3: MyStruct, bar3: ti.types.NDArray[ti.i32, 1]) -> None:
        z3[24] += 89
        my_struct3.a[46] += 32
        my_struct3.b[48] += 33
        my_struct3.c[42] += 34
        bar3[112] += 125
        s3(z3=z3, my_struct3=my_struct3, bar3=bar3)

    @ti.func
    def s1(z2: ti.types.NDArray[ti.i32, 1], my_struct2: MyStruct, bar2: ti.types.NDArray[ti.i32, 1]) -> None:
        z2[22] += 88
        my_struct2.a[45] += 22
        my_struct2.b[47] += 23
        my_struct2.c[41] += 24
        bar2[111] += 123
        s2(z3=z2, my_struct3=my_struct2, bar3=bar2)

    @ti.kernel
    def k1(z: ti.types.NDArray[ti.i32, 1], my_struct: MyStruct, bar: ti.types.NDArray[ti.i32, 1]) -> None:
        z[33] += 2
        my_struct.a[35] += 3
        my_struct.b[37] += 5
        my_struct.c[51] += 17
        bar[222] = 41
        s1(z2=z, my_struct2=my_struct, bar2=bar)

    my_struct = MyStruct(a=a, b=b, c=c)
    k1(z=d, my_struct=my_struct, bar=e)
    assert d[33] == 2
    assert a[35] == 3
    assert b[37] == 5
    assert c[51] == 17

    assert d[22] == 88
    assert a[45] == 22
    assert b[47] == 23
    assert c[41] == 24
    assert e[111] == 123

    assert d[24] == 89
    assert a[46] == 32
    assert b[48] == 33
    assert c[42] == 34
    assert e[112] == 125

    assert d[25] == 90
    assert a[47] == 42
    assert b[49] == 43
    assert c[43] == 44
    assert e[113] == 125

    assert a[1] == 888
    assert b[2] == 999


@test_utils.test()
@pytest.mark.parametrize("use_ndarray", [False, True])
def test_ndarray_struct(ti_type: Any, ti_annotation: Any) -> None:
    gc.collect()
    gc.collect()
    a = ti_type(ti.i32, shape=(55,))
    b = ti_type(ti.i32, shape=(57, 23))
    c = ti_type(ti.i32, shape=(211, 34, 25))
    d = ti_type(ti.i32, shape=(223,))
    e = ti_type(ti.i32, shape=(227,))

    @dataclass
    class MyStruct:
        a: ti_annotation[ti.i32, 1]
        b: ti_annotation[ti.i32, 2]
        c: ti_annotation[ti.i32, 3]

    @ti.func
    def s3(z3: ti_annotation[ti.i32, 1], my_struct3: MyStruct, bar3: ti_annotation[ti.i32, 1]) -> None:
        # stores
        z3[25] += 90
        my_struct3.a[47] += 42
        my_struct3.b[49, 0] += 43
        my_struct3.c[43, 0, 0] += 44
        bar3[113] += 125

        # loads
        bar3[16] = z3[1]
        my_struct3.a[17] = z3[1]
        my_struct3.b[18, 0] = my_struct3.a[3]
        my_struct3.c[19, 0, 0] = my_struct3.b[18, 0]
        z3[20] = my_struct3.c[5, 0, 0]

    @ti.func
    def s2(z3: ti_annotation[ti.i32, 1], my_struct3: MyStruct, bar3: ti_annotation[ti.i32, 1]) -> None:
        # stores
        z3[24] += 89
        my_struct3.a[46] += 32
        my_struct3.b[48, 0] += 33
        my_struct3.c[42, 0, 0] += 34
        bar3[112] += 125
        s3(z3, my_struct3, bar3)

    @ti.func
    def s1(z2: ti_annotation[ti.i32, 1], my_struct2: MyStruct, bar2: ti_annotation[ti.i32, 1]) -> None:
        # stores
        z2[22] += 88
        my_struct2.a[45] += 22
        my_struct2.b[47, 0] += 23
        my_struct2.c[41, 0, 0] += 24
        bar2[111] += 123
        s2(z2, my_struct2, bar2)

    @ti.kernel
    def k1(z: ti_annotation[ti.i32, 1], my_struct: MyStruct, bar: ti_annotation[ti.i32, 1]) -> None:
        # stores
        z[33] += 2
        my_struct.a[35] += 3
        my_struct.b[37, 0] += 5
        my_struct.c[51, 0, 0] += 17
        bar[222] = 41

        # loads
        bar[6] = z[1]
        my_struct.a[7] = z[1]
        my_struct.b[8, 0] = my_struct.a[3]
        my_struct.c[9, 0, 0] = my_struct.b[8, 0]
        z[10] = my_struct.c[5, 0, 0]
        s1(z, my_struct, bar)

    d[1] = 11
    a[3] = 12
    b[2, 0] = 13
    c[5, 0, 0] = 14
    e[4] = 15

    my_struct = MyStruct(a=a, b=b, c=c)
    k1(d, my_struct, e)
    # store tests k1
    assert d[33] == 2
    assert a[35] == 3
    assert b[37, 0] == 5
    assert c[51, 0, 0] == 17

    # from load tests, k1
    assert e[6] == 11
    assert a[7] == 11
    assert b[8, 0] == 12
    assert c[9, 0, 0] == 12
    assert d[10] == 14

    assert d[22] == 88
    assert a[45] == 22
    assert b[47, 0] == 23
    assert c[41, 0, 0] == 24
    assert e[111] == 123

    assert d[24] == 89
    assert a[46] == 32
    assert b[48, 0] == 33
    assert c[42, 0, 0] == 34
    assert e[112] == 125

    # s3 stores
    assert d[25] == 90
    assert a[47] == 42
    assert b[49, 0] == 43
    assert c[43, 0, 0] == 44
    assert e[113] == 125

    # s3 loads
    assert e[16] == 11
    assert a[17] == 11
    assert b[18, 0] == 12
    assert c[19, 0, 0] == 12
    assert d[20] == 14


@test_utils.test()
def test_ndarray_struct_diverse_params():
    gc.collect()
    gc.collect()

    a = ti.ndarray(ti.i32, shape=(55,))
    b = ti.ndarray(ti.i32, shape=(57,))
    c = ti.ndarray(ti.i32, shape=(211,))
    z_param = ti.ndarray(ti.i32, shape=(223,))
    bar_param = ti.ndarray(ti.i32, shape=(227,))

    field1 = ti.field(ti.i32, shape=(300,))

    @dataclass
    class MyStructAB:
        a: ti.types.NDArray[ti.i32, 1]
        b: ti.types.NDArray[ti.i32, 1]

    @dataclass
    class MyStructC:
        c: ti.types.NDArray[ti.i32, 1]

    @ti.func
    def s2(
        my_struct_ab3: MyStructAB,
        z3: ti.types.NDArray[ti.i32, 1],
        fieldparam1_3: ti.template(),
        my_struct_c3: MyStructC,
        bar3: ti.types.NDArray[ti.i32, 1],
    ) -> None:
        # stores
        z3[24] += 89
        my_struct_ab3.a[46] += 32
        my_struct_ab3.b[48] += 33
        my_struct_c3.c[42] += 34
        bar3[112] += 125
        fieldparam1_3[4] = 69

    @ti.func
    def s1(
        z2: ti.types.NDArray[ti.i32, 1],
        my_struct_c2: MyStructC,
        my_struct_ab2: MyStructAB,
        fieldparam1_2: ti.template(),
        bar2: ti.types.NDArray[ti.i32, 1],
    ) -> None:
        # stores
        z2[22] += 88
        my_struct_ab2.a[45] += 22
        my_struct_ab2.b[47] += 23
        my_struct_c2.c[41] += 24
        bar2[111] += 123
        fieldparam1_2[3] = 68

        s2(my_struct_ab2, z2, fieldparam1_2, my_struct_c2, bar2)

    @ti.kernel
    def k1(
        z: ti.types.NDArray[ti.i32, 1],
        my_struct_ab: MyStructAB,
        bar: ti.types.NDArray[ti.i32, 1],
        my_struct_c: MyStructC,
        fieldparam1: ti.template(),
    ) -> None:
        # stores
        z[33] += 2
        my_struct_ab.a[35] += 3
        my_struct_ab.b[37] += 5
        my_struct_c.c[51] += 17
        bar[222] = 41
        fieldparam1[2] = 67

        # loads
        bar[6] = z[1]
        my_struct_ab.a[7] = z[1]
        my_struct_ab.b[8] = my_struct_ab.a[3]
        my_struct_c.c[9] = my_struct_ab.b[8]
        z[10] = my_struct_c.c[5]
        bar[7] = fieldparam1[3]

        s1(z, my_struct_c, my_struct_ab, fieldparam1, bar)

    z_param[1] = 11
    a[3] = 12
    b[2] = 13
    c[5] = 14
    bar_param[4] = 15
    field1[3] = 16

    my_struct_ab_param = MyStructAB(a=a, b=b)
    my_struct_c_param = MyStructC(c=c)
    k1(z_param, my_struct_ab_param, bar_param, my_struct_c_param, field1)
    # store tests k1
    assert z_param[33] == 2
    assert a[35] == 3
    assert b[37] == 5
    assert c[51] == 17
    assert bar_param[222] == 41
    assert field1[2] == 67

    # from load tests, k1
    assert bar_param[6] == 11
    assert a[7] == 11
    assert b[8] == 12
    assert c[9] == 12
    assert z_param[10] == 14
    assert bar_param[7] == 16

    # s1
    assert z_param[22] == 88
    assert a[45] == 22
    assert b[47] == 23
    assert c[41] == 24
    assert bar_param[111] == 123
    assert field1[3] == 68

    # s2
    assert z_param[24] == 89
    assert a[46] == 32
    assert b[48] == 33
    assert c[42] == 34
    assert bar_param[112] == 125
    assert field1[4] == 69


@test_utils.test()
@pytest.mark.parametrize("use_ndarray", [False, True])
def test_ndarray_struct_primitives(ti_type: Any, ti_annotation: Any) -> None:
    gc.collect()
    gc.collect()

    a = ti_type(ti.i32, shape=(55,))
    b = ti_type(ti.i32, shape=(57,))
    c = ti_type(ti.i32, shape=(211,))
    z_param = ti_type(ti.i32, shape=(223,))
    bar_param = ti_type(ti.i32, shape=(227,))

    @dataclass
    class MyStructAB:
        p3: ti.i32
        a: ti_annotation[ti.i32, 1]
        p1: ti.i32
        p2: ti.i32

    @dataclass
    class MyStructC:
        c: ti_annotation[ti.i32, 1]

    @ti.kernel
    def k1(
        z: ti_annotation[ti.i32, 1],
        my_struct_ab: MyStructAB,
        bar: ti_annotation[ti.i32, 1],
        my_struct_c: MyStructC,
    ) -> None:
        my_struct_ab.a[36] += my_struct_ab.p1
        my_struct_ab.a[37] += my_struct_ab.p2
        my_struct_ab.a[38] += my_struct_ab.p3

    my_struct_ab_param = MyStructAB(a=a, p1=119, p2=123, p3=345)
    my_struct_c_param = MyStructC(c=c)
    k1(z_param, my_struct_ab_param, bar_param, my_struct_c_param)
    assert a[36] == 119
    assert a[37] == 123
    assert a[38] == 345


@test_utils.test()
def test_ndarray_struct_nested_ndarray():
    a = ti.ndarray(ti.i32, shape=(55,))
    b = ti.ndarray(ti.i32, shape=(57,))
    c = ti.ndarray(ti.i32, shape=(211,))
    d = ti.ndarray(ti.i32, shape=(211,))
    e = ti.ndarray(ti.i32, shape=(251,))
    f = ti.ndarray(ti.i32, shape=(251,))

    @dataclass
    class MyStructEF:
        e: ti.types.NDArray[ti.i32, 1]
        f: ti.types.NDArray[ti.i32, 1]

    @dataclass
    class MyStructCD:
        c: ti.types.NDArray[ti.i32, 1]
        d: ti.types.NDArray[ti.i32, 1]
        struct_ef: MyStructEF

    @dataclass
    class MyStructAB:
        a: ti.types.NDArray[ti.i32, 1]
        b: ti.types.NDArray[ti.i32, 1]
        struct_cd: MyStructCD

    @ti.func
    def f3(
        my_struct_ab3: MyStructAB,
    ) -> None:
        my_struct_ab3.a[47] += 23
        my_struct_ab3.b[42] += 25
        my_struct_ab3.struct_cd.c[51] += 33
        my_struct_ab3.struct_cd.d[57] += 43
        my_struct_ab3.struct_cd.struct_ef.e[52] += 34
        my_struct_ab3.struct_cd.struct_ef.f[58] += 44

        my_struct_ab3.a[50] = my_struct_ab3.a.shape[0]
        my_struct_ab3.a[51] = my_struct_ab3.struct_cd.c.shape[0]
        my_struct_ab3.a[52] = my_struct_ab3.struct_cd.struct_ef.e.shape[0]

    @ti.func
    def f2(
        my_struct_ab2: MyStructAB,
    ) -> None:
        my_struct_ab2.a[27] += 13
        my_struct_ab2.b[22] += 15
        my_struct_ab2.struct_cd.c[31] += 23
        my_struct_ab2.struct_cd.d[37] += 33
        my_struct_ab2.struct_cd.struct_ef.e[32] += 24
        my_struct_ab2.struct_cd.struct_ef.f[38] += 34
        f3(my_struct_ab2)
        my_struct_ab2.a[60] = my_struct_ab2.a.shape[0]
        my_struct_ab2.a[61] = my_struct_ab2.struct_cd.c.shape[0]
        my_struct_ab2.a[62] = my_struct_ab2.struct_cd.struct_ef.e.shape[0]

    @ti.kernel
    def k1(
        my_struct_ab: MyStructAB,
    ) -> None:
        my_struct_ab.a[7] += 3
        my_struct_ab.b[2] += 5
        my_struct_ab.struct_cd.c[11] += 13
        my_struct_ab.struct_cd.d[17] += 23
        my_struct_ab.struct_cd.struct_ef.e[12] += 14
        my_struct_ab.struct_cd.struct_ef.f[18] += 24
        f2(my_struct_ab)
        my_struct_ab.a[70] = my_struct_ab.a.shape[0]
        my_struct_ab.a[71] = my_struct_ab.struct_cd.c.shape[0]
        my_struct_ab.a[72] = my_struct_ab.struct_cd.struct_ef.e.shape[0]

    my_struct_ef_param = MyStructEF(e=e, f=f)
    my_struct_cd_param = MyStructCD(c=c, d=d, struct_ef=my_struct_ef_param)
    my_struct_ab_param = MyStructAB(a=a, b=b, struct_cd=my_struct_cd_param)
    k1(my_struct_ab_param)

    assert a[7] == 3
    assert b[2] == 5
    assert c[11] == 13
    assert d[17] == 23
    assert e[12] == 14
    assert f[18] == 24

    assert a[27] == 13
    assert b[22] == 15
    assert c[31] == 23
    assert d[37] == 33
    assert e[32] == 24
    assert f[38] == 34

    assert a[47] == 23
    assert b[42] == 25
    assert c[51] == 33
    assert d[57] == 43
    assert e[52] == 34
    assert f[58] == 44

    # shapes
    assert a[50] == 55
    assert a[51] == 211
    assert a[52] == 251

    assert a[60] == 55
    assert a[61] == 211
    assert a[62] == 251

    assert a[70] == 55
    assert a[71] == 211
    assert a[72] == 251


@test_utils.test()
def test_field_struct_nested_field() -> None:
    a = ti.field(ti.i32, shape=(55,))
    b = ti.field(ti.i32, shape=(57,))
    c = ti.field(ti.i32, shape=(211,))
    d = ti.field(ti.i32, shape=(211,))
    e = ti.field(ti.i32, shape=(251,))
    f = ti.field(ti.i32, shape=(251,))

    @dataclass
    class MyStructEF:
        e: ti.Template
        f: ti.Template

    @dataclass
    class MyStructCD:
        c: ti.Template
        d: ti.Template
        struct_ef: MyStructEF

    @dataclass
    class MyStructAB:
        a: ti.Template
        b: ti.Template
        struct_cd: MyStructCD

    @ti.func
    def f3(
        my_struct_ab3: MyStructAB,
    ) -> None:
        my_struct_ab3.a[47] += 23
        my_struct_ab3.b[42] += 25
        my_struct_ab3.struct_cd.c[51] += 33
        my_struct_ab3.struct_cd.d[57] += 43
        my_struct_ab3.struct_cd.struct_ef.e[52] += 34
        my_struct_ab3.struct_cd.struct_ef.f[58] += 44
        my_struct_ab3.a[50] = my_struct_ab3.a.shape[0]
        my_struct_ab3.a[51] = my_struct_ab3.struct_cd.c.shape[0]
        my_struct_ab3.a[52] = my_struct_ab3.struct_cd.struct_ef.e.shape[0]

    @ti.func
    def f2(
        my_struct_ab2: MyStructAB,
    ) -> None:
        my_struct_ab2.a[27] += 13
        my_struct_ab2.b[22] += 15
        my_struct_ab2.struct_cd.c[31] += 23
        my_struct_ab2.struct_cd.d[37] += 33
        my_struct_ab2.struct_cd.struct_ef.e[32] += 24
        my_struct_ab2.struct_cd.struct_ef.f[38] += 34
        f3(my_struct_ab2)
        my_struct_ab2.a[60] = my_struct_ab2.a.shape[0]
        my_struct_ab2.a[61] = my_struct_ab2.struct_cd.c.shape[0]
        my_struct_ab2.a[62] = my_struct_ab2.struct_cd.struct_ef.e.shape[0]

    @ti.kernel
    def k1(
        my_struct_ab: MyStructAB,
    ) -> None:
        my_struct_ab.a[7] += 3
        my_struct_ab.b[2] += 5
        my_struct_ab.struct_cd.c[11] += 13
        my_struct_ab.struct_cd.d[17] += 23
        my_struct_ab.struct_cd.struct_ef.e[12] += 14
        my_struct_ab.struct_cd.struct_ef.f[18] += 24
        f2(my_struct_ab)
        my_struct_ab.a[70] = my_struct_ab.a.shape[0]
        my_struct_ab.a[71] = my_struct_ab.struct_cd.c.shape[0]
        my_struct_ab.a[72] = my_struct_ab.struct_cd.struct_ef.e.shape[0]

    my_struct_ef_param = MyStructEF(e=e, f=f)
    my_struct_cd_param = MyStructCD(c=c, d=d, struct_ef=my_struct_ef_param)
    my_struct_ab_param = MyStructAB(a=a, b=b, struct_cd=my_struct_cd_param)
    k1(my_struct_ab_param)

    assert a[7] == 3
    assert b[2] == 5
    assert c[11] == 13
    assert d[17] == 23
    assert e[12] == 14
    assert f[18] == 24

    assert a[27] == 13
    assert b[22] == 15
    assert c[31] == 23
    assert d[37] == 33
    assert e[32] == 24
    assert f[38] == 34

    assert a[47] == 23
    assert b[42] == 25
    assert c[51] == 33
    assert d[57] == 43
    assert e[52] == 34
    assert f[58] == 44

    # shapes
    assert a[50] == 55
    assert a[51] == 211
    assert a[52] == 251

    assert a[60] == 55
    assert a[61] == 211
    assert a[62] == 251

    assert a[70] == 55
    assert a[71] == 211
    assert a[72] == 251


@test_utils.test()
def test_ndarray_struct_multiple_child_structs_ndarray():
    a = ti.ndarray(ti.i32, shape=(55,))
    b = ti.ndarray(ti.i32, shape=(57,))
    c = ti.ndarray(ti.i32, shape=(211,))
    d = ti.ndarray(ti.i32, shape=(211,))
    e = ti.ndarray(ti.i32, shape=(251,))
    f = ti.ndarray(ti.i32, shape=(251,))

    d11 = ti.ndarray(ti.i32, shape=(251,))
    d12 = ti.ndarray(ti.i32, shape=(251,))
    d21 = ti.ndarray(ti.i32, shape=(251,))
    d22 = ti.ndarray(ti.i32, shape=(251,))
    d31 = ti.ndarray(ti.i32, shape=(251,))
    d32 = ti.ndarray(ti.i32, shape=(251,))

    @dataclass
    class D1:
        d11: ti.types.NDArray[ti.i32, 1]
        d12: ti.types.NDArray[ti.i32, 1]

    @dataclass
    class D2:
        d21: ti.types.NDArray[ti.i32, 1]
        d22: ti.types.NDArray[ti.i32, 1]

    @dataclass
    class D3:
        d31: ti.types.NDArray[ti.i32, 1]
        d32: ti.types.NDArray[ti.i32, 1]

    @dataclass
    class C1:
        a: ti.types.NDArray[ti.i32, 1]
        d1: D1
        d2: D2
        d3: D3
        b: ti.types.NDArray[ti.i32, 1]

    @dataclass
    class C2:
        c: ti.types.NDArray[ti.i32, 1]
        d: ti.types.NDArray[ti.i32, 1]

    @dataclass
    class C3:
        e: ti.types.NDArray[ti.i32, 1]
        f: ti.types.NDArray[ti.i32, 1]

    @dataclass
    class P1:
        c1: C1
        c2: C2
        c3: C3

    @ti.kernel
    def k1(p1: P1) -> None:
        p1.c1.a[0] = 22
        p1.c1.b[0] = 33
        p1.c2.c[0] = 44
        p1.c2.d[0] = 55
        p1.c3.e[0] = 66
        p1.c3.f[0] = 77

    d1 = D1(d11=d11, d12=d12)
    d2 = D2(d21=d21, d22=d22)
    d3 = D3(d31=d31, d32=d32)
    c1 = C1(a=a, b=b, d1=d1, d2=d2, d3=d3)
    c2 = C2(c=c, d=d)
    c3 = C3(e=e, f=f)
    p1 = P1(c1=c1, c2=c2, c3=c3)
    k1(p1)
    assert a[0] == 22
    assert b[0] == 33
    assert c[0] == 44
    assert d[0] == 55
    assert e[0] == 66
    assert f[0] == 77


@test_utils.test()
def test_ndarray_struct_multiple_child_structs_field():
    a = ti.field(ti.i32, shape=(55,))
    b = ti.field(ti.i32, shape=(57,))
    c = ti.field(ti.i32, shape=(211,))
    d = ti.field(ti.i32, shape=(211,))
    e = ti.field(ti.i32, shape=(251,))
    f = ti.field(ti.i32, shape=(251,))

    @dataclass
    class C1:
        a: ti.Template
        b: ti.Template

    @dataclass
    class C2:
        c: ti.Template
        d: ti.Template

    @dataclass
    class C3:
        e: ti.Template
        f: ti.Template

    @dataclass
    class P1:
        c1: C1
        c2: C2
        c3: C3

    @ti.kernel
    def k1(p1: P1) -> None:
        p1.c1.a[0] = 22
        p1.c1.b[0] = 33
        p1.c2.c[0] = 44
        p1.c2.d[0] = 55
        p1.c3.e[0] = 66
        p1.c3.f[0] = 77

    c1 = C1(a=a, b=b)
    c2 = C2(c=c, d=d)
    c3 = C3(e=e, f=f)
    p1 = P1(c1=c1, c2=c2, c3=c3)
    k1(p1)
    assert a[0] == 22
    assert b[0] == 33
    assert c[0] == 44
    assert d[0] == 55
    assert e[0] == 66
    assert f[0] == 77
