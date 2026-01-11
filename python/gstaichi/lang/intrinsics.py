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


def clock_speed_hz():
    """
    Returns the clock speed in Hz corresponding to the clock counter.

    This function runs on the host side and queries the device for its clock rate.
    The returned value can be used to convert clock cycles from clock_counter() to time.

    Supported backends:
    - CUDA: Returns the GPU clock rate in Hz

    Unsupported backends (returns 0.0):
    - AMDGPU: Returns 0.0
    - Vulkan: Returns 0.0
    - Metal: Returns 0.0
    - CPU: Returns 0.0

    Returns:
        float: Clock rate in Hz, or 0.0 if not supported

    Example::

        >>> import gstaichi as ti
        >>> ti.init(arch=ti.cuda)
        >>> clock_rate_hz = ti.clock_speed_hz()
        >>> print(f"GPU clock rate: {clock_rate_hz / 1e9:.2f} GHz")
        >>> 
        >>> # Use with clock_counter to measure time
        >>> @ti.kernel
        >>> def timed_kernel() -> ti.f64:
        >>>     start = ti.clock_counter()
        >>>     # ... do work ...
        >>>     end = ti.clock_counter()
        >>>     return (end - start) / clock_rate_hz  # time in seconds
    """
    arch = impl.get_runtime().prog.config().arch
    if arch == _ti_core.cuda:
        # query_int64 returns kHz, convert to Hz
        clock_rate_khz = _ti_core.query_int64("cuda_clock_rate_khz")
        return float(clock_rate_khz * 1000)
    # Return 0.0 for unsupported backends
    return 0.0


__all__ = [
    "clock_counter",
    "clock_speed_hz",
]
