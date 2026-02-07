import inspect
import time
from collections import defaultdict
from typing import Any, Callable, Generic, ParamSpec, TypeVar

from . import impl
from ._gstaichi_callable import GsTaichiCallable
from .exception import GsTaichiRuntimeError, GsTaichiSyntaxError

NUM_WARMUP: int = 2


class DispatchKernelImpl:
    kernel_impl_idx: int = 1

    def __init__(self, kernel: GsTaichiCallable, is_compatible: Callable | None) -> None:
        self.is_compatible: Callable | None = is_compatible
        self.__wrapped__: GsTaichiCallable = kernel
        self.kernel_impl_idx = DispatchKernelImpl.kernel_impl_idx
        DispatchKernelImpl.kernel_impl_idx += 1

    def __call__(self, *args, **kwargs) -> Any:
        return self.__wrapped__(*args, **kwargs)


P = ParamSpec("P")
R = TypeVar("R")


class PerformanceDispatcher(Generic[P, R]):
    def __init__(self, get_geometry_hash: Callable[P, int], fn: Callable, num_warmup: int | None = None) -> None:
        self.num_warmup = num_warmup if num_warmup else NUM_WARMUP
        sig = inspect.signature(fn)
        self._param_types: dict[str, Any] = {}
        for param_name, param in sig.parameters.items():
            self._param_types[param_name] = param.annotation
        self._get_geometry_hash: Callable[P, int] = get_geometry_hash
        self._kernel_by_idx: dict[int, DispatchKernelImpl] = {}
        self._trial_count_by_kernel_idx_by_geometry_hash: dict[int, dict[int, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        self._fastest_by_geometry_hash: dict[int, DispatchKernelImpl] = defaultdict(None)
        self._times_by_kernel_idx_by_geometry_hash: dict[int, dict[int, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )

    def register(
        self, kernel: GsTaichiCallable | None = None, *, is_compatible: Callable[[dict], bool] | None = None
    ) -> Callable[[GsTaichiCallable], GsTaichiCallable]:
        """
        Use register to register a @ti.kernel with a @ti.perf_dispatch meta kernel

        See @ti.perf_dispatch for documentation about using @ti.perf_dispatch meta kernels

        is_compatible is an optional function that will return whether the kernel being registered can
        run on the specific arguments being passed in. If there are circumstances where this kernel being
        registered cannot run, then is_compatible MUST be implemented, and MUST return False given the specific arguments
        or platform.

        is_compatible receives the exact same *args and **kwargs that were used to call the meta kernel.

        Examples of where you might need to implement is_compatible:
        - the kernel only runs on Metal => is_compatible should return False on any platform where Metal is not
          available (typically, any non-Darwin machine for example)
        - the kernel only runs for certain ranges of dimensions on one or more of the input arguments
            - in this case, check the shape of the argument in question, and return False if out of spec for this
              kernel implementation
        """
        kernel_by_idx = self._kernel_by_idx

        def decorator(func: GsTaichiCallable) -> GsTaichiCallable:
            if not type(func) in {GsTaichiCallable}:
                raise GsTaichiSyntaxError("@ti.perf_dispatch should be placed before @ti.kernel")
            sig = inspect.signature(func.fn)
            for param_name, _param in sig.parameters.items():
                if param_name not in self._param_types:
                    raise GsTaichiSyntaxError(
                        f"Signature parameter {param_name} of kernel not in perf_dispatch function prototype"
                    )
            if len(sig.parameters) != len(self._param_types):
                raise GsTaichiSyntaxError(
                    f"Number of kernel parameters {len(sig.parameters)} doesn't match number of parameters in perf_dispatch function prototype {len(self._param_types)}"
                )

            dispatch_impl = DispatchKernelImpl(kernel=func, is_compatible=is_compatible)
            kernel_by_idx[dispatch_impl.kernel_impl_idx] = dispatch_impl
            return func

        if kernel is not None:
            return decorator(kernel)
        return decorator

    def _get_compatible_kernels(self, *args, **kwargs) -> dict[int, DispatchKernelImpl]:
        compatible = {}
        for kernel_idx, kernel in self._kernel_by_idx.items():
            if kernel.is_compatible and not kernel.is_compatible(*args, **kwargs):
                continue
            compatible[kernel_idx] = kernel
        return compatible

    def _get_next_kernel_idx(self, compatible: dict[int, DispatchKernelImpl], geometry_hash: int) -> int:
        least_trials_idx = None
        least_trials = None
        for kernel_idx in compatible.keys():
            trial_count = self._trial_count_by_kernel_idx_by_geometry_hash[geometry_hash].get(kernel_idx, 0)
            if least_trials is None or trial_count < least_trials:
                least_trials_idx = kernel_idx
                least_trials = trial_count
        assert least_trials_idx is not None
        return least_trials_idx

    def _get_finished_trials(self, geometry_hash: int) -> bool:
        return min(self._trial_count_by_kernel_idx_by_geometry_hash[geometry_hash].values()) >= self.num_warmup + 1

    def _update_fastest(self, geometry_hash: int) -> None:
        speeds_l = []
        for kernel_idx, elapsed_time in self._times_by_kernel_idx_by_geometry_hash[geometry_hash].items():
            speeds_l.append((kernel_idx, elapsed_time))
        speeds_l.sort(key=lambda x: x[1], reverse=False)
        self._fastest_by_geometry_hash[geometry_hash] = self._kernel_by_idx[speeds_l[0][0]]

    def __call__(self, *args: P.args, **kwargs: P.kwargs):
        """
        We are going to run each kernel self.num_warmup times, to warm up, then run them each again,
        then choose the fastest kernel, based on the time of the last run.

        Each kernel must have identical behavior, including for side-effects.

        We only run a single kernel per call, so kernels don't need to be idempotent.

        We call sync before and after, because kernels run async, so:
        - if we didn't sync after, we'd measure the time to queue the kernel, without waiting for it to finish.
        - if we didn't sync before, we'd be measuring also the time for all the existing gpu kernels that
          have already been queued up, are processing. So we sync to make sure those have finished first.

        We collect a single sample from each implementation, and compare that single sample with the samples from the
        other implementations.

        We are comparing algorithms based on empirical runtime.

        Note that for best results, sets of input arguments that have different runtimes should map to different
        geometries, otherwise the comparison between runtimes might not be fair, and an inappropriate implementation
        kernel might be selected.

        We are not implementing an epsilon-greedy algorithm to keep sampling non-fastest variants just in case the
        distribution is shifting over time.

        It is not possible for you to control exploration vs exploitation.
        """
        geometry_hash = self._get_geometry_hash(*args, **kwargs)
        fastest = self._fastest_by_geometry_hash.get(geometry_hash)
        if fastest:
            return fastest(*args, **kwargs)

        res = None
        speeds_l = []
        runtime = impl.get_runtime()
        compatible = self._get_compatible_kernels(*args, **kwargs)
        if len(compatible) == 0:
            raise GsTaichiRuntimeError("No suitable kernels were found.")

        elif len(compatible) == 1:
            self._fastest_by_geometry_hash[geometry_hash] = next(iter(compatible.values()))
            return self._fastest_by_geometry_hash[geometry_hash](*args, **kwargs)

        kernel_idx = self._get_next_kernel_idx(compatible=compatible, geometry_hash=geometry_hash)
        kernel = self._kernel_by_idx[kernel_idx]
        runtime.sync()
        start = time.time()
        res = kernel(*args, **kwargs)
        runtime.sync()
        end = time.time()
        elapsed = end - start
        speeds_l.append((elapsed, kernel))
        trial_count_by_kernel_idx = self._trial_count_by_kernel_idx_by_geometry_hash[geometry_hash]
        trial_count_by_kernel_idx[kernel_idx] += 1
        if trial_count_by_kernel_idx[kernel_idx] >= self.num_warmup:
            self._times_by_kernel_idx_by_geometry_hash[geometry_hash][kernel_idx].append(elapsed)
        if self._get_finished_trials(geometry_hash=geometry_hash):
            self._update_fastest(geometry_hash)
            speeds_l.sort(key=lambda x: x[0], reverse=False)
            self._fastest_by_geometry_hash[geometry_hash] = speeds_l[0][1]
        return res


def perf_dispatch(*, get_geometry_hash: Callable):
    """
    This annotation designates a meta-kernel that can have one or more @ti.kernel's registered with it.

    At runtime, gstaichi will try running each registered kernel in turn, and choose the fastest. Once
    chosen, the fastest kernel will systematically be used, for the lifetime of the process. This is
    aimed for use where there are multiple possible kernel implementations, and no clear heuristic to
    choose between them.

    Example usage:

    @ti.perf_dispatch(get_geometry_hash=lambda a, c: hash(a.shape + c.shape))
    def my_func1(a: ti.types.NDArray[ti.i32, 1], c: ti.types.NDArray[ti.i32, 1]): ...
        # note: this is intentionally empty. The function body will NEVER be called.

    @my_func1.register
    @ti.kernel
    def my_func1_impl1(a: ti.types.NDArray[ti.i32, 1], c: ti.types.NDArray[ti.i32, 1]) -> None:
        # implementation 1 here...

    @my_func1.register(is_compatible=lambda a, c: a.shape[0] < 2)
    @ti.kernel
    def my_func1_impl2(a: ti.types.NDArray[ti.i32, 1], c: ti.types.NDArray[ti.i32, 1]) -> None:
        # implementation 2 here...

    Then simply call the meta-kernel, just like any other kernel:

    my_func1(a, b)

    Note that the effect of each implementation must be identical, including side effects, otherwise subtle
    and hard to diagnose bugs are likely to occur. @ti.perf_dispatch does NOT check that the implementations have
    identical effects.

    ## Geometry

    Depending on certain characteristics of the input arguments to a call, different implementations might be
    relatively faster or slower. We denote such characteristics the 'geometry' of the call. An example of 'geometry'
    is the stride and padding to a call to a convolutional kernel, as well as the number of channels, the height
    and the width.

    The meta kernel @ti.perf_dispatch annotation MUST provide a function that returns a geometry hash
    given the arguments.

    You are free to return any valid hash.
    - In the simplest case, you could simply return a constant value, in which case all inputs will be considered to
      have identical 'geometry', and the same implemnetation kernel will systematically be called
    - Otherwise, if you are aware of key characteristics of the input arguments, then you can return a hash of these
      characteristics here

    Note that it is strongly recommended that any values used to create the geometry hash are NOT retrieved from data
    on the GPU, otherwise you are likely to create a GPU sync point, which would be likely to severely slow down
    performance.

    Examples of geometry could be simply the shapes of all input arguments:

    get_geometry_hash=lambda *args, **kwargs: hash(tuple([arg.shape for arg in args]))

    ### Advanced geometry

    You can simply hash the input arguments directly:

    get_geometry_hash=lambda *args, **kwargs: hash(tuple(*args, frozendict(kwargs)))
    """

    def decorator(fn: GsTaichiCallable):
        return PerformanceDispatcher(get_geometry_hash=get_geometry_hash, fn=fn)

    return decorator


__all__ = ["perf_dispatch"]
