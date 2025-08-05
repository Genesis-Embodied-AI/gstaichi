# type: ignore

from gstaichi.lang import impl


def memfence():
    return impl.call_internal("grid_memfence", with_runtime_context=False)
