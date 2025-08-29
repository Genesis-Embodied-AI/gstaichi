import pathlib

import gstaichi as ti
from gstaichi._test_tools import ti_init_same_arch

from tests import test_utils


@test_utils.test()
def test_fe_ll_observations(tmp_path: pathlib.Path) -> None:
    @ti.kernel
    def k1(a: ti.types.NDArray[ti.i32, 1]) -> None:
        a[0] += 1

    ti_init_same_arch(offline_cache_file_path=str(tmp_path), offline_cache=True)
    a = ti.ndarray(ti.i32, (10,))
    assert not k1._primal.fe_ll_cache_observations.cache_hit
    k1(a)
    assert not k1._primal.fe_ll_cache_observations.cache_hit

    ti_init_same_arch(offline_cache_file_path=str(tmp_path), offline_cache=True)
    a = ti.ndarray(ti.i32, (10,))
    k1._primal.fe_ll_cache_observations.cache_hit = False
    k1(a)
    assert k1._primal.fe_ll_cache_observations.cache_hit

    ti_init_same_arch(offline_cache_file_path=str(tmp_path), offline_cache=True)
    a = ti.ndarray(ti.i32, (10,))
    k1._primal.fe_ll_cache_observations.cache_hit = False
    k1(a)
    assert k1._primal.fe_ll_cache_observations.cache_hit
