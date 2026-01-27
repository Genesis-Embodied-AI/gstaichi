import time
from collections import defaultdict
from typing import Callable

from . import impl
from ._gstaichi_callable import GsTaichiCallable

NUM_WARMUP = 2


class DispatchKernelImpl:
    def __init__(self, underlying: GsTaichiCallable, is_compatible: Callable | None) -> None:
        self.is_compatible: Callable | None = is_compatible
        self._underlying: GsTaichiCallable = underlying

    @property
    def underlying(self) -> GsTaichiCallable:
        return self._underlying


class KernelSpeedChecker:
    def __init__(self, get_geometry_hash: Callable) -> None:
        self._get_geometry_hash: Callable = get_geometry_hash
        self._underlyings: list[DispatchKernelImpl] = []
        self._trial_count_by_geometry_hash: dict[int, int] = defaultdict(int)
        self._fastest_by_geometry_hash: dict[int, DispatchKernelImpl] = defaultdict(None)

    def register(
        self, kernel: GsTaichiCallable | None = None, *, is_compatible: Callable[[dict], bool] | None = None
    ) -> Callable[[GsTaichiCallable], GsTaichiCallable]:
        def decorator(func: GsTaichiCallable) -> GsTaichiCallable:
            dispatch_impl = DispatchKernelImpl(underlying=func, is_compatible=is_compatible)
            self._underlyings.append(dispatch_impl)
            return func

        if kernel is not None:
            return decorator(kernel)
        return decorator

    def __call__(self, *args, **kwargs):
        geometry_hash = self._get_geometry_hash(*args, **kwargs)
        fastest = self._fastest_by_geometry_hash.get(geometry_hash)
        if fastest:
            return fastest.underlying(*args, **kwargs)

        res = None
        speeds_l = []
        runtime = impl.get_runtime()
        for underlying in self._underlyings:
            if underlying.is_compatible and not underlying.is_compatible(*args, **kwargs):
                continue
            runtime.sync()
            start = time.time()
            res = underlying.underlying(*args, **kwargs)
            runtime.sync()
            end = time.time()
            elapsed = end - start
            speeds_l.append((elapsed, underlying))
        self._trial_count_by_geometry_hash[geometry_hash] += 1
        if self._trial_count_by_geometry_hash[geometry_hash] > NUM_WARMUP:
            speeds_l.sort(key=lambda x: x[0], reverse=False)
            self._fastest_by_geometry_hash[geometry_hash] = speeds_l[0][1]
        return res


def perf_dispatch(*, get_geometry_hash: Callable):
    def decorator(fn: GsTaichiCallable):
        speed_checker = KernelSpeedChecker(get_geometry_hash=get_geometry_hash)
        return speed_checker

    return decorator


__all__ = ["perf_dispatch"]
