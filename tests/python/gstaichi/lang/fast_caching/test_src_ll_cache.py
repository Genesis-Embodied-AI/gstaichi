import pathlib

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


# Should be enough to run these on cpu I think, and anything involving
# stdout/stderr capture is fairly flaky on other arch
@test_utils.test(arch=ti.cpu)
def test_src_ll_cache_arg_warnings(tmp_path: pathlib.Path, capfd) -> None:
    ti_init_same_arch(offline_cache_file_path=str(tmp_path), offline_cache=True)

    class RandomClass:
        pass

    @ti.pure
    @ti.kernel
    def k1(foo: ti.Template) -> None:
        pass

    k1(foo=RandomClass())
    _out, err = capfd.readouterr()
    err += _out  # Windows doesnt have stderr
    assert "FASTCACHE_PARAM_INVALID" in err
    assert RandomClass.__name__ in err
    assert "FASTCACHE_INVALID_FUNC" in err
    assert k1.__name__ in err

    @ti.kernel
    def not_pure_k1(foo: ti.Template) -> None:
        pass

    not_pure_k1(foo=RandomClass())
    _out, err = capfd.readouterr()
    err += _out  # Windows doesnt have stderr
    assert "FASTCACHE_PARAM_INVALID" not in err
    assert RandomClass.__name__ not in err
    assert "FASTCACHE_INVALID_FUNC" not in err
    assert k1.__name__ not in err
