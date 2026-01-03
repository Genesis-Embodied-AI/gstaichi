# type: ignore

from gstaichi._lib import core as _ti_core
from gstaichi.lang import impl


def clock_counter():
    """
    Returns the current value of a hardware cycle counter.

    All backends return raw clock cycles or ticks, NOT nanoseconds.
    The counter frequency varies by hardware and may change dynamically
    (e.g., due to GPU boost or thermal throttling).

    Supported backends:
    - CUDA: Per-streaming-multiprocessor cycle counter (increments every SM clock cycle).
      See https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#time-function
    - AMDGPU: GPU cycle counter
    - Vulkan: Device clock in cycles (requires VK_KHR_shader_clock, else returns 0)
    - CPU (x64/arm64): Processor timestamp counter (constant rate on modern CPUs)

    Unsupported backends (returns 0):
    - Metal

    Use this for relative timing measurements within the same backend and run.
    Comparing cycle counts across different backends or hardware is not meaningful.
    """
    arch = impl.get_runtime().prog.config().arch
    if arch == _ti_core.cuda:
        return impl.call_internal("cuda_clock_i64", with_runtime_context=False)
    if arch == _ti_core.amdgpu:
        return impl.call_internal("amdgpu_clock_i64", with_runtime_context=False)
    if arch == _ti_core.vulkan:
        return impl.call_internal("spirv_clock_i64", with_runtime_context=False)
    if arch == _ti_core.x64 or arch == _ti_core.arm64:
        return impl.call_internal("cpu_clock_i64", with_runtime_context=False)
    # No-op if not supported
    return 0


__all__ = [
    "clock_counter",
]
