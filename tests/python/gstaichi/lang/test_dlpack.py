import pytest
from tests import test_utils
import gstaichi as ti
import torch


dlpack_arch = [ti.cpu, ti.cuda]
dlpack_ineligible_arch = [ti.metal, ti.vulkan]


@test_utils.test(arch=dlpack_arch)
@pytest.mark.parametrize(
    "dtype", [ti.i32, ti.i64, ti.f32, ti.f64, ti.u1]
)
@pytest.mark.parametrize(
    "shape,poses", [
        ((), [()]),
        ((3,), [(0,), (2,)]),
        ((3, 2), [(0, 0), (2, 1), (1, 1)]),
        ((3, 1, 2), [(2, 0, 1), (0, 0, 1)]),
    ]
)
def test_ndarray_dlpack_types(dtype, shape: tuple[int], poses: list[tuple[int, ...]]) -> None:
    ndarray = ti.ndarray(dtype, shape)
    for i, pos in enumerate(poses):
        ndarray[pos] = i * 10 + 10
    dlpack = ndarray.to_dlpack()
    tt = torch.utils.dlpack.from_dlpack(dlpack)
    assert tuple(tt.shape) == shape
    expected_torch_type = {
        ti.i32: torch.int32,
        ti.i64: torch.int64,
        ti.f32: torch.float32,
        ti.f64: torch.float64,
        ti.u1: torch.bool
    }[dtype]
    assert tt.dtype == expected_torch_type
    for i, pos in enumerate(poses):
        assert tt[pos] == ndarray[pos]
        assert tt[pos] != 0


@test_utils.test(arch=dlpack_arch)
def test_ndarray_dlpack_mem_stays_alloced() -> None:
    def create_tensor(shape, dtype):
        nd = ti.ndarray(dtype, shape)
        tt = torch.utils.dlpack.from_dlpack(nd.to_dlpack())
        return tt
    t = create_tensor((3, 2), ti.i32)
    # will crash if memory already deleted
    assert t[0, 0] == 0


@test_utils.test(arch=dlpack_ineligible_arch)
def test_refuses_ineligible_arch() -> None:
    def create_tensor(shape, dtype):
        nd = ti.ndarray(dtype, shape)
        tt = torch.utils.dlpack.from_dlpack(nd.to_dlpack())
        return tt
    with pytest.raises(RuntimeError):
        t = create_tensor((3, 2), ti.i32)
        assert t[0, 0]
