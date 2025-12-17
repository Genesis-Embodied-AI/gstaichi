import gstaichi as ti
from gstaichi.lang.misc import get_host_arch_list

from tests import test_utils


@test_utils.test(arch=get_host_arch_list())
def test_cache_primitive_args():
    @ti.data_oriented
    class StructStaticConfig:
        flag_1: bool = True

    @ti.kernel
    def fun(static_args: ti.template(), flag_2: ti.template(), value: ti.types.ndarray()):
        if ti.static(static_args.flag_1):
            if ti.static(flag_2):
                value[None] = value[None] + 1
            else:
                assert "Invalid 'flag_2' branch"
        else:
            assert "Invalid 'static_args.flag_1' branch"

    assert len(fun._primal.mapper.mapping) == 0
    assert len(fun._primal._launch_ctx_cache_tracker) == 0

    static_args = StructStaticConfig()
    flag_2 = True
    value = ti.ndarray(ti.i32, shape=())
    value[None] = 1

    fun(static_args, flag_2, value)
    assert value[None] == 2
    assert len(fun._primal.mapper.mapping) == 1
    assert len(fun._primal._launch_ctx_cache_tracker) == 1

    fun(static_args, flag_2, value)
    assert value[None] == 3
    assert len(fun._primal.mapper.mapping) == 1
    assert len(fun._primal._launch_ctx_cache_tracker) == 1


@test_utils.test(arch=get_host_arch_list())
def test_cache_fields_only():
    @ti.kernel
    def fun(flag: ti.template(), value: ti.template()):
        if ti.static(flag):
            value[None] = value[None] + 1
        else:
            assert "Invalid 'static_args.flag_1' branch"

    assert len(fun._primal.mapper.mapping) == 0
    assert len(fun._primal._launch_ctx_cache_tracker) == 0

    flag = True
    value = ti.field(ti.i32, shape=())
    value[None] = 1

    fun(flag, value)
    assert value[None] == 2
    assert len(fun._primal.mapper.mapping) == 1
    assert len(fun._primal._launch_ctx_cache_tracker) == 1

    fun(flag, value)
    assert value[None] == 3
    assert len(fun._primal.mapper.mapping) == 1
    assert len(fun._primal._launch_ctx_cache_tracker) == 1
