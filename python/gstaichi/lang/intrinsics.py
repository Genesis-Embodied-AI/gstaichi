# type: ignore

from gstaichi._lib import core as _ti_core
from gstaichi.lang import impl


def clock():
    """
    Returns the value of a hardware counter that is incremented every clock cycle.

    Supported backends:
    - CUDA: Per-streaming-multiprocessor cycle counter.
      See https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#time-function
    - AMDGPU: GPU timestamp counter
    - Vulkan: Device clock (requires VK_KHR_shader_clock support, else returns 0)
    - CPU (x64/arm64): Processor timestamp counter

    Unsupported backends (returns 0):
    - Metal

    Note: The counter frequency and semantics may vary across backends.
    Use this for relative timing measurements within the same backend.
    """
    arch = impl.get_runtime().prog.config().arch
    if arch == _ti_core.cuda:
        return impl.call_internal("cuda_clock_i64", with_runtime_context=False)
    elif arch == _ti_core.amdgpu:
        return impl.call_internal("amdgpu_clock_i64", with_runtime_context=False)
    elif arch == _ti_core.vulkan:
        return impl.call_internal("spirv_clock_i64", with_runtime_context=False)
    elif arch == _ti_core.x64 or arch == _ti_core.arm64:
        return impl.call_internal("cpu_clock_i64", with_runtime_context=False)
    # No-op if not supported
    return 0


__all__ = [
    "clock",
]
