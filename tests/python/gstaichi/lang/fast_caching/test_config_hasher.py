import gstaichi as ti
from gstaichi.lang._fast_caching import config_hasher

from tests import test_utils


@test_utils.test()
def test_config_hasher():
    assert ti.cfg is not None

    ti.init(arch=getattr(ti, ti.cfg.arch.name))
    h_base = config_hasher.hash_compile_config()

    ti.init(arch=getattr(ti, ti.cfg.arch.name))
    h_same = config_hasher.hash_compile_config()

    ti.init(arch=getattr(ti, ti.cfg.arch.name), offline_cache_max_size_of_files=123)
    h_diff = config_hasher.hash_compile_config()

    assert h_base == h_same
    assert h_base != h_diff
