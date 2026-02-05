import time
from collections import defaultdict
from typing import Callable

from . import impl
from ._gstaichi_callable import GsTaichiCallable
from .exception import GsTaichiSyntaxError


NUM_WARMUP: int = 2


class DispatchKernelImpl:
    kernel_impl_idx: int = 1

    def __init__(self, underlying: GsTaichiCallable, is_compatible: Callable | None) -> None:
        if not type(underlying) in {GsTaichiCallable}:
            raise GsTaichiSyntaxError("@ti.perf_dispatch should be placed before @ti.kernel")
        self.is_compatible: Callable | None = is_compatible
        self._underlying: GsTaichiCallable = underlying
        self.kernel_impl_idx = DispatchKernelImpl.kernel_impl_idx
        DispatchKernelImpl.kernel_impl_idx += 1

    @property
    def underlying(self) -> GsTaichiCallable:
        return self._underlying


class KernelSpeedChecker:
    def __init__(self, get_geometry_hash: Callable) -> None:
        self._get_geometry_hash: Callable = get_geometry_hash
        self._underlying_by_idx: dict[int, DispatchKernelImpl] = {}
        self._trial_count_by_underlying_idx_by_geometry_hash: dict[int, dict[int, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        self._fastest_by_geometry_hash: dict[int, DispatchKernelImpl] = defaultdict(None)
        self._times_by_underlying_idx_by_geometry_hash: dict[int, dict[int, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )

    def register(
        self, kernel: GsTaichiCallable | None = None, *, is_compatible: Callable[[dict], bool] | None = None
    ) -> Callable[[GsTaichiCallable], GsTaichiCallable]:
        def decorator(func: GsTaichiCallable) -> GsTaichiCallable:
            dispatch_impl = DispatchKernelImpl(underlying=func, is_compatible=is_compatible)
            self._underlying_by_idx[dispatch_impl.kernel_impl_idx] = dispatch_impl
            return func

        if kernel is not None:
            return decorator(kernel)
        return decorator

    def _get_compatible_underlyings(self, *args, **kwargs) -> dict[int, DispatchKernelImpl]:
        compatible = {}
        for underlying_idx, underlying in self._underlying_by_idx.items():
            if underlying.is_compatible and not underlying.is_compatible(*args, **kwargs):
                continue
            compatible[underlying_idx] = underlying
        return compatible

    def _pick_next_underlying_idx(self, compatible: dict[int, DispatchKernelImpl], geometry_hash: int) -> int:
        least_trials_idx = None
        least_trials = None
        for underlying_idx in compatible.keys():
            trial_count = self._trial_count_by_underlying_idx_by_geometry_hash[geometry_hash].get(underlying_idx, 0)
            if least_trials is None or trial_count < least_trials:
                least_trials_idx = underlying_idx
                least_trials = trial_count
        assert least_trials_idx is not None
        return least_trials_idx

    def _finished_trials(self, geometry_hash: int) -> bool:
        return (
            min(self._trial_count_by_underlying_idx_by_geometry_hash[geometry_hash].values())
            >= NUM_WARMUP + 1
        )

    def _calculate_fastest(self, geometry_hash: int) -> None:
        speeds_l = []
        for underlying_idx, elapsed_time in self._times_by_underlying_idx_by_geometry_hash[geometry_hash].items():
            speeds_l.append((underlying_idx, elapsed_time))
        speeds_l.sort(key=lambda x: x[1], reverse=False)
        self._fastest_by_geometry_hash[geometry_hash] = self._underlying_by_idx[speeds_l[0][0]]

    def __call__(self, *args, **kwargs):
        geometry_hash = self._get_geometry_hash(*args, **kwargs)
        fastest = self._fastest_by_geometry_hash.get(geometry_hash)
        if fastest:
            return fastest.underlying(*args, **kwargs)

        res = None
        speeds_l = []
        runtime = impl.get_runtime()
        compatible = self._get_compatible_underlyings(*args, **kwargs)
        underlying_idx = self._pick_next_underlying_idx(compatible=compatible, geometry_hash=geometry_hash)
        underlying = self._underlying_by_idx[underlying_idx]
        runtime.sync()
        start = time.time()
        res = underlying.underlying(*args, **kwargs)
        runtime.sync()
        end = time.time()
        elapsed = end - start
        speeds_l.append((elapsed, underlying))
        self._trial_count_by_underlying_idx_by_geometry_hash[geometry_hash][underlying_idx] += 1
        if self._trial_count_by_underlying_idx_by_geometry_hash[geometry_hash][underlying_idx] >= NUM_WARMUP:
            self._times_by_underlying_idx_by_geometry_hash[geometry_hash][underlying_idx].append(elapsed)
        if self._finished_trials(geometry_hash=geometry_hash):
            self._calculate_fastest(geometry_hash)
            speeds_l.sort(key=lambda x: x[0], reverse=False)
            self._fastest_by_geometry_hash[geometry_hash] = speeds_l[0][1]
        return res


def perf_dispatch(*, get_geometry_hash: Callable):
    def decorator(fn: GsTaichiCallable):
        speed_checker = KernelSpeedChecker(get_geometry_hash=get_geometry_hash)
        return speed_checker

    return decorator


__all__ = ["perf_dispatch"]
