import os
import pathlib
import sys
from io import StringIO

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


class StderrCapture:
    def __init__(self, tmp_file):
        self.tmp_file = tmp_file

    def read(self):
        self.tmp_file.flush()
        self.tmp_file.seek(0)
        out = self.tmp_file.read()
        self.tmp_file.truncate(0)
        self.tmp_file.seek(0)
        return out


@pytest.fixture
def redirect_stderr(tmp_path):
    """
    This redirects the underlying operating system stderr. Since
    we are calling into a c++ library that does writes to stderr,
    then redirecting things only in python land is not enough.
    Concretely, using capfd works on linux, and works for stdout
    on Windows, but I couldn't get it working with stderr on Windows.
    """
    tmp_stderr_path = tmp_path / "stderr.txt"
    with open(tmp_stderr_path, "w+") as tmp_stderr:
        old_stderr_fd = os.dup(2)
        os.dup2(tmp_stderr.fileno(), 2)
        capture = StderrCapture(tmp_stderr)
        try:
            yield capture
        finally:
            os.dup2(old_stderr_fd, 2)
            os.close(old_stderr_fd)


# Should be enough to run these on cpu I think, and anything involving
# stdout/stderr capture is fairly flaky on other arch
@test_utils.test(arch=ti.cpu)
def test_src_ll_cache_arg_warnings(tmp_path: pathlib.Path, redirect_stderr) -> None:
    ti_init_same_arch(offline_cache_file_path=str(tmp_path), offline_cache=True)

    class RandomClass:
        pass

    @ti.pure
    @ti.kernel
    def k1(foo: ti.Template) -> None:
        pass

    k1(foo=RandomClass())
    err = redirect_stderr.read()
    assert "FASTCACHE_PARAM_INVALID" in err
    assert RandomClass.__name__ in err
    assert "FASTCACHE_INVALID_FUNC" in err
    assert k1.__name__ in err

    @ti.kernel
    def not_pure_k1(foo: ti.Template) -> None:
        pass

    not_pure_k1(foo=RandomClass())
    err = redirect_stderr.read()
    assert "FASTCACHE_PARAM_INVALID" not in err
    assert RandomClass.__name__ not in err
    assert "FASTCACHE_INVALID_FUNC" not in err
    assert k1.__name__ not in err
