import gstaichi as ti

from tests import test_utils


def _arch_supports_clock(arch):
    """Check if the architecture supports the clock intrinsic."""
    if arch == ti.vulkan:
        # Vulkan: check device capability at runtime
        device_caps = ti.lang.impl.get_runtime().prog.get_device_caps()
        return device_caps.get(ti._lib.core.DeviceCapability.spirv_has_int64) != 0
    # CPU and CUDA/AMDGPU always support int64
    return arch in (ti.cuda, ti.amdgpu, ti.x64, ti.arm64)


@test_utils.test()
def test_clock():
    arch = ti.lang.impl.get_runtime().prog.config().arch

    dtype = ti.i64 if _arch_supports_clock(arch) else ti.i32
    a = ti.field(dtype=dtype, shape=32)

    @ti.kernel
    def foo():
        ti.loop_config(serialize=True, block_dim=1)
        for i in range(32):
            start = ti.clock_counter()
            x = ti.random()
            for j in range((i + 1) * 50000):
                if x > 0.99:
                    x = x / 100
                else:
                    x = ti.sqrt(x)
            if x != 0:
                a[i] = ti.clock_counter() - start

    foo()

    if _arch_supports_clock(arch):
        for i in range(1, 31):
            assert a[i - 1] < a[i] < a[i + 1]
            if arch not in (ti.x64, ti.arm64):
                # CPU clock is time based, not cycle based
                assert -1 < a[i] / a[0] - (i + 1) < 1
    else:
        # On unsupported backends, clock returns 0
        for i in range(1, 31):
            assert a[i] == 0
