# type: ignore

from gstaichi.lang import impl


def sync():
    """Blocks the calling thread until all the previously
    launched GsTaichi kernels have completed.
    """
    impl.get_runtime().sync()


__all__ = ["sync"]
