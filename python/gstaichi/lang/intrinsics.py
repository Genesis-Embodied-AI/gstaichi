# type: ignore

from gstaichi._lib import core as _ti_core
from gstaichi.lang import impl


def clock():
    """
    Returns the value of a per-multiprocessor counter that is incremented every clock cycle.

    It returns 0 for unsupported arch instead of raising an exception since failure is harmless.

    See official CUDA documentation for details:
    https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#time-function
    """
    arch = impl.get_runtime().prog.config().arch
    if arch == _ti_core.cuda:
        return impl.call_internal("cuda_clock_i64", with_runtime_context=False)
    return 0


__all__ = [
    "clock",
]
