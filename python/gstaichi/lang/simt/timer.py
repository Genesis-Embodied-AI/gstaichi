# type: ignore

from gstaichi.lang import impl


def cuda_clock_i64():
    return impl.call_internal("cuda_clock_i64", with_runtime_context=False)


__all__ = [
    "cuda_clock_i64",
]
