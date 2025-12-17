import gstaichi as ti
from gstaichi.lang.misc import get_host_arch_list

from tests import test_utils


@test_utils.test(arch=get_host_arch_list())
def test_cache_primitive_args():
    @ti.data_oriented
    class StructStaticConfig:
        flag: bool = True

    @ti.kernel
    def fun(static_args: ti.template(), constant: ti.template(), value: ti.types.ndarray()):
        if ti.static(static_args.flag):
            if ti.static(constant > 0):
                value[None] = value[None] + 1
            else:
                assert "Invalid 'constant' branch"
        else:
            assert "Invalid 'static_args.flag' branch"

    assert len(fun._primal.mapper._mapping_cache) == 0
    assert len(fun._primal.mapper._mapping_cache_tracker) == 0
    assert len(fun._primal._launch_ctx_cache) == 0
    assert len(fun._primal._launch_ctx_cache_tracker) == 0

    static_args = StructStaticConfig()
    constant = 1234567890
    value = ti.ndarray(ti.i32, shape=())
    value[None] = 1

    fun(static_args, constant, value)
    assert value[None] == 2
    assert len(fun._primal.mapper._mapping_cache) == 1
    assert len(fun._primal.mapper._mapping_cache_tracker) == 1
    assert len(fun._primal._launch_ctx_cache) == 1
    assert len(fun._primal._launch_ctx_cache_tracker) == 1

    constant_2 = int("1234567890")  # Must be smart to defeat object interning and get a different address
    assert id(constant) != id(constant_2)
    fun(static_args, constant_2, value)
    assert value[None] == 3
    assert len(fun._primal.mapper._mapping_cache) == 1
    assert len(fun._primal.mapper._mapping_cache_tracker) == 1
    assert len(fun._primal._launch_ctx_cache) == 1
    assert len(fun._primal._launch_ctx_cache_tracker) == 1

    static_args_2 = StructStaticConfig()
    assert id(static_args) != id(static_args_2)
    fun(static_args_2, constant, value)
    assert value[None] == 4
    assert len(fun._primal.mapper._mapping_cache) == 2
    assert len(fun._primal.mapper._mapping_cache_tracker) == 2
    assert len(fun._primal._launch_ctx_cache) == 2
    assert len(fun._primal._launch_ctx_cache_tracker) == 2


@test_utils.test(arch=get_host_arch_list())
def test_cache_multi_entry_static():
    @ti.kernel
    def fun(flag: ti.template(), value: ti.template()):
        if ti.static(flag):
            value[None] = value[None] + 1
        else:
            value[None] = value[None] - 1

    assert len(fun._primal.mapper._mapping_cache) == 0
    assert len(fun._primal.mapper._mapping_cache_tracker) == 0
    assert len(fun._primal._launch_ctx_cache) == 0
    assert len(fun._primal._launch_ctx_cache_tracker) == 0

    value = ti.field(ti.i32, shape=())
    value[None] = 1

    fun(True, value)
    assert value[None] == 2
    assert len(fun._primal.mapper._mapping_cache) == 1
    assert len(fun._primal.mapper._mapping_cache_tracker) == 1
    assert len(fun._primal._launch_ctx_cache) == 1
    assert len(fun._primal._launch_ctx_cache_tracker) == 1

    fun(True, value)
    assert value[None] == 3
    assert len(fun._primal.mapper._mapping_cache) == 1
    assert len(fun._primal.mapper._mapping_cache_tracker) == 1
    assert len(fun._primal._launch_ctx_cache) == 1
    assert len(fun._primal._launch_ctx_cache_tracker) == 1

    fun(False, value)
    assert value[None] == 2
    assert len(fun._primal.mapper._mapping_cache) == 2
    assert len(fun._primal.mapper._mapping_cache_tracker) == 2
    assert len(fun._primal._launch_ctx_cache) == 2
    assert len(fun._primal._launch_ctx_cache_tracker) == 2



@test_utils.test(arch=get_host_arch_list())
def test_cache_fields_only():
    @ti.kernel
    def fun(flag: ti.template(), value: ti.template()):
        if ti.static(flag):
            value[None] = value[None] + 1
        else:
            assert "Invalid 'static_args.flag_1' branch"

    assert len(fun._primal.mapper._mapping_cache) == 0
    assert len(fun._primal.mapper._mapping_cache_tracker) == 0
    assert len(fun._primal._launch_ctx_cache) == 0
    assert len(fun._primal._launch_ctx_cache_tracker) == 0

    flag = True
    value = ti.field(ti.i32, shape=())
    value[None] = 1

    fun(flag, value)
    assert value[None] == 2
    assert len(fun._primal.mapper._mapping_cache) == 1
    assert len(fun._primal.mapper._mapping_cache_tracker) == 1
    assert len(fun._primal._launch_ctx_cache) == 1
    assert len(fun._primal._launch_ctx_cache_tracker) == 1

    fun(flag, value)
    assert value[None] == 3
    assert len(fun._primal.mapper._mapping_cache) == 1
    assert len(fun._primal.mapper._mapping_cache_tracker) == 1
    assert len(fun._primal._launch_ctx_cache) == 1
    assert len(fun._primal._launch_ctx_cache_tracker) == 1
