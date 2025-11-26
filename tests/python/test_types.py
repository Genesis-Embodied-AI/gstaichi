import pytest
import numpy as np

import gstaichi as ti
from gstaichi.lang import impl

from tests import test_utils

_TI_TYPES = [ti.i8, ti.i16, ti.i32, ti.u8, ti.u16, ti.u32, ti.f32]
_TI_64_TYPES = [ti.i64, ti.u64, ti.f64]


def _test_type_assign_argument(dt):
    x = ti.field(dt, shape=())

    @ti.kernel
    def func(value: dt):
        x[None] = value

    func(3)
    assert x[None] == 3


@pytest.mark.parametrize("dt", _TI_TYPES)
@test_utils.test(exclude=[ti.vulkan])
def test_type_assign_argument(dt):
    _test_type_assign_argument(dt)


@pytest.mark.parametrize("dt", _TI_64_TYPES)
@test_utils.test(exclude=[ti.vulkan], require=ti.extension.data64)
def test_type_assign_argument64(dt):
    _test_type_assign_argument(dt)


def _test_type_operator(dt):
    x = ti.field(dt, shape=())
    y = ti.field(dt, shape=())
    add = ti.field(dt, shape=())
    mul = ti.field(dt, shape=())

    @ti.kernel
    def func():
        add[None] = x[None] + y[None]
        mul[None] = x[None] * y[None]

    for i in range(0, 3):
        for j in range(0, 3):
            x[None] = i
            y[None] = j
            func()
            assert add[None] == x[None] + y[None]
            assert mul[None] == x[None] * y[None]


@pytest.mark.parametrize("dt", _TI_TYPES)
@test_utils.test(exclude=[ti.vulkan])
def test_type_operator(dt):
    _test_type_operator(dt)


@pytest.mark.parametrize("dt", _TI_64_TYPES)
@test_utils.test(exclude=[ti.vulkan], require=ti.extension.data64)
def test_type_operator64(dt):
    _test_type_operator(dt)


def _test_type_field(dt):
    x = ti.field(dt, shape=(3, 2))

    @ti.kernel
    def func(i: ti.i32, j: ti.i32):
        x[i, j] = 3

    for i in range(0, 3):
        for j in range(0, 2):
            func(i, j)
            assert x[i, j] == 3


@pytest.mark.parametrize("dt", _TI_TYPES)
@test_utils.test(exclude=[ti.vulkan])
def test_type_field(dt):
    _test_type_field(dt)


@pytest.mark.parametrize("dt", _TI_64_TYPES)
@test_utils.test(exclude=[ti.vulkan], require=ti.extension.data64)
def test_type_field64(dt):
    _test_type_field(dt)


def _test_overflow(dt, n):
    a = ti.field(dt, shape=())
    b = ti.field(dt, shape=())
    c = ti.field(dt, shape=())

    @ti.kernel
    def func():
        c[None] = a[None] + b[None]

    a[None] = 2**n // 3
    b[None] = 2**n // 3

    func()

    assert a[None] == 2**n // 3
    assert b[None] == 2**n // 3

    if ti.types.is_signed(dt):
        assert c[None] == 2**n // 3 * 2 - (2**n)  # overflows
    else:
        assert c[None] == 2**n // 3 * 2  # does not overflow


@pytest.mark.parametrize(
    "dt,n",
    [
        (ti.i8, 8),
        (ti.u8, 8),
        (ti.i16, 16),
        (ti.u16, 16),
        (ti.i32, 32),
        (ti.u32, 32),
    ],
)
@test_utils.test(exclude=[ti.vulkan])
def test_overflow(dt, n):
    _test_overflow(dt, n)


@pytest.mark.parametrize(
    "dt,n",
    [
        (ti.i64, 64),
        (ti.u64, 64),
    ],
)
@test_utils.test(exclude=[ti.vulkan], require=ti.extension.data64)
def test_overflow64(dt, n):
    _test_overflow(dt, n)


@pytest.mark.parametrize(
    "dt,val",
    [
        (ti.u32, 0xFFFFFFFF),
        (ti.u64, 0xFFFFFFFFFFFFFFFF),
    ],
)
@test_utils.test(require=ti.extension.data64)
def test_uint_max(dt, val):
    # https://github.com/taichi-dev/gstaichi/issues/2060
    impl.get_runtime().default_ip = dt
    N = 16
    f = ti.field(dt, shape=N)

    @ti.kernel
    def run():
        for i in f:
            f[i] = val

    run()
    fs = f.to_numpy()
    for f in fs:
        assert f == val


@pytest.mark.parametrize("tensor_type", [ti.field, ti.ndarray])
@pytest.mark.parametrize("dtype", [ti.u1, ti.u8, ti.u16, ti.u32, ti.u64, ti.i8, ti.i32, ti.i16, ti.i64])
@test_utils.test()
def test_types_fields_and_dtypes_kernel_write_to_numpy_consistency(tensor_type, dtype) -> None:
    """
    Check consistency between:
    - write elements in kernel
    - to numpy
    """
    poses = [0, 2, 5, 11]
    a = tensor_type(dtype, (16,))

    TensorType = ti.types.NDArray if tensor_type == ti.ndarray else ti.Template

    @ti.kernel
    def k1(a: TensorType) -> None:
        for b_ in range(1):
            for pos in ti.static(poses):
                a[pos] = 1

    k1(a)

    a_np = a.to_numpy()

    for i in range(16):
        assert a_np[i] == (1 if i in poses else 0)


@pytest.mark.parametrize("tensor_type", [ti.field, ti.ndarray])
@pytest.mark.parametrize("dtype", [ti.u1, ti.u8, ti.u16, ti.u32, ti.u64, ti.i8, ti.i32, ti.i16, ti.i64])
@test_utils.test()
def test_types_fields_and_dtypes_kernel_from_numpy_to_numpy_consistency(tensor_type, dtype) -> None:
    """
    Check consistency between:
    - from numpy
    - to numpy
    """
    poses = [0, 2, 5, 11]

    ti_dtype_to_np_dtype = {
        ti.u1: np.bool_,
        ti.u8: np.uint8,
        ti.u16: np.uint16,
        ti.u32: np.uint32,
        ti.u64: np.uint64,
        ti.i8: np.int8,
        ti.i16: np.int16,
        ti.i32: np.int32,
        ti.i64: np.int64
    }
    np_dtype = ti_dtype_to_np_dtype[dtype]
    
    a_np = np.zeros(dtype=np_dtype, shape=(16,))

    for pos in poses:
        a_np[pos] = 1

    a = tensor_type(dtype, (16,))
    a.from_numpy(a_np)

    b_np = a.to_numpy()

    for i in range(16):
        assert b_np[i] == (1 if i in poses else 0)
