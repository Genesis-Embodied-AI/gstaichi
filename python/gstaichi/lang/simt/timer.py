# type: ignore

from gstaichi.lang import impl


def cuda_clock_i64():
    """
    Returns the value of a per-multiprocessor counter that is incremented every clock cycle.

    See official documentation for details:
    https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#time-function
    """
    return impl.call_internal("cuda_clock_i64", with_runtime_context=False)


__all__ = [
    "cuda_clock_i64",
]
