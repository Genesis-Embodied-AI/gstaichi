import pytest
import torch

import gstaichi as ti

from tests import test_utils

dlpack_arch = [ti.cpu, ti.cuda]
dlpack_ineligible_arch = [ti.metal, ti.vulkan]


def ti_to_torch(ti_tensor: ti.types.NDArray) -> torch.Tensor:
    cap = ti_tensor.to_dlpack()
    torch_tensor = torch.utils.dlpack.from_dlpack(cap)
    return torch_tensor


@test_utils.test(arch=dlpack_arch)
@pytest.mark.parametrize("tensor_type", [ti.ndarray, ti.field])
@pytest.mark.parametrize("dtype", [ti.i32, ti.i64, ti.f32, ti.f64, ti.u1])
@pytest.mark.parametrize(
    "shape,poses",
    [
        ((), [()]),
        ((3,), [(0,), (2,)]),
        ((3, 2), [(0, 0), (2, 1), (1, 1)]),
        ((3, 1, 2), [(2, 0, 1), (0, 0, 1)]),
    ],
)
def test_dlpack_types(tensor_type, dtype, shape: tuple[int], poses: list[tuple[int, ...]]) -> None:
    ti_tensor = tensor_type(dtype, shape)
    for i, pos in enumerate(poses):
        ti_tensor[pos] = i * 10 + 10
    dlpack = ti_tensor.to_dlpack()
    tt = torch.utils.dlpack.from_dlpack(dlpack)
    assert tuple(tt.shape) == shape
    expected_torch_type = {
        ti.i32: torch.int32,
        ti.i64: torch.int64,
        ti.f32: torch.float32,
        ti.f64: torch.float64,
        ti.u1: torch.bool,
    }[dtype]
    assert tt.dtype == expected_torch_type
    for i, pos in enumerate(poses):
        assert tt[pos] == ti_tensor[pos]
        assert tt[pos] != 0


@test_utils.test(arch=dlpack_arch)
def test_dlpack_ndarray_mem_stays_alloced() -> None:
    """
    On fields, memory always stays allocated till ti.reset(), so we
    don't need to check. Not true with ndarrays.
    """

    def create_tensor(shape, dtype):
        nd = ti.ndarray(dtype, shape)
        tt = torch.utils.dlpack.from_dlpack(nd.to_dlpack())
        return tt

    t = create_tensor((3, 2), ti.i32)
    # will crash if memory already deleted
    assert t[0, 0] == 0


@test_utils.test(arch=dlpack_ineligible_arch)
@pytest.mark.parametrize("tensor_type", [ti.ndarray, ti.field])
def test_dlpack_refuses_ineligible_arch(tensor_type) -> None:
    def create_tensor(shape, dtype):
        nd = tensor_type(dtype, shape)
        tt = torch.utils.dlpack.from_dlpack(nd.to_dlpack())
        return tt

    with pytest.raises(RuntimeError):
        t = create_tensor((3, 2), ti.i32)
        t[0, 0]


@test_utils.test(arch=dlpack_arch)
@pytest.mark.parametrize("tensor_type", [ti.ndarray, ti.field])
def test_dlpack_vec3(tensor_type):
    vec3 = ti.types.vector(3, ti.f32)
    a = tensor_type(vec3, shape=(10, 3))
    a[0, 0] = (5, 4, 3)
    a[0, 1] = (7, 8, 9)
    a[1, 0] = (11, 12, 13)
    tt = ti_to_torch(a)
    assert tuple(tt.shape) == (10, 3, 3)
    assert tt.dtype == torch.float32
    assert tt[0, 0, 0] == 5
    assert tt[0, 0, 1] == 4
    assert tt[0, 0, 2] == 3
    assert tt[0, 1, 0] == 7
    assert tt[0, 1, 1] == 8
    assert tt[0, 1, 2] == 9
    assert tt[1, 0, 0] == 11
    assert tt[1, 0, 1] == 12
    assert tt[1, 0, 2] == 13


@test_utils.test(arch=dlpack_arch)
@pytest.mark.parametrize("tensor_type", [ti.ndarray, ti.field])
def test_dlpack_mat2x3(tensor_type):
    vec3 = ti.types.matrix(2, 3, ti.f32)
    a = tensor_type(vec3, shape=(10, 3))
    a[0, 0] = ((5, 4, 1), (3, 2, 20))
    a[0, 1] = ((7, 8, 21), (9, 10, 22))
    a[1, 0] = ((11, 12, 23), (13, 14, 23))
    tt = ti_to_torch(a)
    assert tuple(tt.shape) == (10, 3, 2, 3)
    assert tt.dtype == torch.float32
    assert tt[0, 0, 0, 0] == 5
    assert tt[0, 0, 0, 1] == 4
    assert tt[0, 0, 0, 2] == 1
    assert tt[0, 0, 1, 0] == 3
    assert tt[0, 0, 1, 1] == 2
    assert tt[0, 0, 1, 2] == 20
    assert tt[0, 1, 1, 1] == 10


@test_utils.test(arch=dlpack_arch)
@pytest.mark.parametrize("tensor_type", [ti.ndarray, ti.field])
def test_dlpack_2_arrays(tensor_type):
    """
    Just in case we need to handle offset
    """
    a = tensor_type(ti.i32, (100,))
    b = tensor_type(ti.i32, (100,))
    a[0] = 123
    b[0] = 222
    a_t = ti_to_torch(a)
    b_t = ti_to_torch(b)
    assert a_t[0] == 123
    assert b_t[0] == 222


@test_utils.test(arch=dlpack_arch)
def test_dlpack_field_multiple_tree_nodes():
    """
    each ti.sync causes the fields to be written to a new snode tree node
    each tree node has its own memory block, so we want to test:
    - multiple snodes within same tree node
    - different tree nodes
    ... just to check we aren't aliasing somehow in the to_dlpack function
    """
    tensor_type = ti.field
    a = tensor_type(ti.i32, (100,))
    b = tensor_type(ti.i32, (100,))
    ti.sync()
    c = tensor_type(ti.i32, (100,))
    ti.sync()
    d = tensor_type(ti.i32, (100,))
    e = tensor_type(ti.i32, (100,))
    ti.sync()

    # check the tree node ids
    assert a.snode._snode_tree_id == 0
    assert b.snode._snode_tree_id == 0
    assert c.snode._snode_tree_id == 1
    assert d.snode._snode_tree_id == 2
    assert e.snode._snode_tree_id == 2

    a[0] = 123
    b[0] = 222
    c[0] = 333
    d[0] = 444
    e[0] = 555

    a_t = ti_to_torch(a)
    b_t = ti_to_torch(b)
    c_t = ti_to_torch(c)
    d_t = ti_to_torch(d)
    e_t = ti_to_torch(e)

    assert a_t[0] == 123
    assert b_t[0] == 222
    assert c_t[0] == 333
    assert d_t[0] == 444
    assert e_t[0] == 555
