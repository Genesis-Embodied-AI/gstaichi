import pytest
from tests import test_utils
import gstaichi as ti
import torch


@test_utils.test()
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
    # "shape", [(3,), (3, 2), (3, 1, 2)]
)
def test_ndarray_dlpack(dtype, shape: tuple[int], poses: list[tuple[int, ...]]) -> None:
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
