import pathlib
import pytest

import gstaichi as ti
from gstaichi._lib.core import gstaichi_python
from gstaichi._test_tools import ti_init_same_arch

from tests import test_utils


@test_utils.test()
def test_src_ll_cache1(tmp_path: pathlib.Path) -> None:
    ti_init_same_arch(offline_cache_file_path=str(tmp_path), offline_cache=True)

    @ti.kernel
    def no_pure() -> None:
        pass

    no_pure()
    assert not no_pure._primal.src_ll_cache_observations.cache_key_generated

    ti_init_same_arch(offline_cache_file_path=str(tmp_path), offline_cache=True)

    @ti.pure
    @ti.kernel
    def has_pure() -> None:
        pass

    has_pure()
    assert has_pure._primal.src_ll_cache_observations.cache_key_generated
    assert not has_pure._primal.src_ll_cache_observations.cache_validated
    assert not has_pure._primal.src_ll_cache_observations.cache_loaded
    assert has_pure._primal.src_ll_cache_observations.cache_stored

    ti_init_same_arch(offline_cache_file_path=str(tmp_path), offline_cache=True)

    has_pure()
    assert has_pure._primal.src_ll_cache_observations.cache_key_generated
    assert has_pure._primal.src_ll_cache_observations.cache_validated
    assert has_pure._primal.src_ll_cache_observations.cache_loaded


@test_utils.test()
@pytest.mark.parametrize(
    "flag_value", [
        None,
        False,
        True,
    ]
)
def test_src_ll_cache_flag(tmp_path: pathlib.Path, flag_value: bool) -> None:
    """
    Test ti.init(src_ll_cache) flag
    """
    if flag_value:
        ti_init_same_arch(offline_cache_file_path=str(tmp_path), src_ll_cache=flag_value)
    else:
        ti_init_same_arch()
    @ti.pure
    @ti.kernel
    def k1() -> None:
        pass

    k1()
    cache_used = k1._primal.src_ll_cache_observations.cache_key_generated
    if flag_value:
        assert cache_used == flag_value
    else:
        assert cache_used  # default
