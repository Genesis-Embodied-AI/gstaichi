# Must import 'partial' directly instead of the entire module to avoid attribute lookup overhead.
from functools import update_wrapper
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from .kernel import Kernel


class GsTaichiCallable:
    """
    BoundGsTaichiCallable is used to enable wrapping a bindable function with a class.

    Design requirements for GsTaichiCallable:
    - wrap/contain a reference to a class Func instance, and allow (the GsTaichiCallable) being passed around
      like normal function pointer
    - expose attributes of the wrapped class Func, such as `_if_real_function`, `_primal`, etc
    - allow for (now limited) strong typing, and enable type checkers, such as pyright/mypy
        - currently GsTaichiCallable is a shared type used for all functions marked with @ti.func, @ti.kernel,
          python functions (?)
        - note: current type-checking implementation does not distinguish between different type flavors of
          GsTaichiCallable, with different values of `_if_real_function`, `_primal`, etc
    - handle not only class-less functions, but also class-instance methods (where determining the `self`
      reference is a challenge)

    Let's take the following example:

    def test_ptr_class_func():
    @ti.data_oriented
    class MyClass:
        def __init__(self):
            self.a = ti.field(dtype=ti.f32, shape=(3))

        def add2numbers_py(self, x, y):
            return x + y

        @ti.func
        def add2numbers_func(self, x, y):
            return x + y

        @ti.kernel
        def func(self):
            a, add_py, add_func = ti.static(self.a, self.add2numbers_py, self.add2numbers_func)
            a[0] = add_py(2, 3)
            a[1] = add_func(3, 7)

    (taken from test_ptr_assign.py).

    When the @ti.func decorator is parsed, the function `add2numbers_func` exists, but there is not yet any `self`
    - it is not possible for the method to be bound, to a `self` instance
    - however, the @ti.func annotation, runs the kernel_imp.py::func function --- it is at this point
      that GsTaichi's original code creates a class Func instance (that wraps the add2numbers_func)
      and immediately we create a GsTaichiCallable instance that wraps the Func instance.
    - effectively, we have two layers of wrapping GsTaichiCallable->Func->function pointer
      (actual function definition)
    - later on, when we call self.add2numbers_py, here:

            a, add_py, add_func = ti.static(self.a, self.add2numbers_py, self.add2numbers_func)

      ... we want to call the bound method, `self.add2numbers_py`.
    - an actual python function reference, created by doing somevar = MyClass.add2numbers, can automatically
      binds to self, when called from self in this way (however, add2numbers_py is actually a class
      Func instance, wrapping python function reference -- now also all wrapped by a GsTaichiCallable
      instance -- returned by the kernel_impl.py::func function, run by @ti.func)
    - however, in order to be able to add strongly typed attributes to the wrapped python function, we need
      to wrap the wrapped python function in a class
    - the wrapped python function, wrapped in a GsTaichiCallable class (which is callable, and will
      execute the underlying double-wrapped python function), will NOT automatically bind
    - when we invoke GsTaichiCallable, the wrapped function is invoked. The wrapped function is unbound, and
      so `self` is not automatically passed in, as an argument, and things break

    To address this we need to use the `__get__` method, in our function wrapper, ie GsTaichiCallable,
    and have the `__get__` method return the `BoundGsTaichiCallable` object. The `__get__` method handles
    running the binding for us, and effectively binds `BoundFunc` object to `self` object, by passing
    in the instance, as an argument into `BoundGsTaichiCallable.__init__`.

    `BoundFunc` can then be used as a normal bound func - even though it's just an object instance -
    using its `__call__` method. Effectively, at the time of actually invoking the underlying python
    function, we have 3 layers of wrapper instances:
        BoundGsTaichiCallabe -> GsTaichiCallable -> Func -> python function reference/definition
    """

    def __init__(self, fn: Callable, wrapper: Callable) -> None:
        self.fn: Callable = fn
        self.wrapper: Callable = wrapper
        self._is_real_function: bool = False
        self._is_gstaichi_function: bool = False
        self._is_wrapped_kernel: bool = False
        self._is_classkernel: bool = False
        self._primal: "Kernel | None" = None
        self._adjoint: "Kernel | None" = None
        self.grad: "Kernel | None" = None
        self.is_pure: bool = False
        update_wrapper(self, fn)

    def __call__(self, *args, **kwargs):
        return self.wrapper.__call__(*args, **kwargs)

    def __get__(self, instance, owner):
        if instance is None:
            return self
        return BoundGsTaichiCallable(instance, self)


class BoundGsTaichiCallable:
    def __init__(self, instance: Any, gstaichi_callable: GsTaichiCallable):
        self.wrapper = gstaichi_callable.wrapper
        self.instance = instance
        self.gstaichi_callable = gstaichi_callable

    def __call__(self, *args, **kwargs):
        return self.wrapper(self.instance, *args, **kwargs)

    def __getattr__(self, k: str) -> Any:
        res = getattr(self.gstaichi_callable, k)
        return res

    def __setattr__(self, k: str, v: Any) -> None:
        # Note: these have to match the name of any attributes on this class.
        if k in {"wrapper", "instance", "gstaichi_callable"}:
            object.__setattr__(self, k, v)
        else:
            setattr(self.gstaichi_callable, k, v)

    def grad(self, *args, **kwargs) -> "Kernel":
        assert self.gstaichi_callable._adjoint is not None
        return self.gstaichi_callable._adjoint(self.instance, *args, **kwargs)
