import ast
import dataclasses
import functools
import inspect
import json
import operator
import os
import pathlib
import re
import sys
import textwrap
import time
import types
import typing
import warnings
from typing import Any, Callable, Type

import numpy as np

import gstaichi.lang
import gstaichi.lang._ndarray
import gstaichi.lang._texture
import gstaichi.types.annotations
from gstaichi import _logging
from gstaichi._lib import core as _ti_core
from gstaichi._lib.core.gstaichi_python import (
    ASTBuilder,
    CompiledKernelData,
    CompileResult,
    FunctionKey,
    KernelCxx,
    KernelLaunchContext,
)
from gstaichi.lang import _kernel_impl_dataclass, impl, ops, runtime_ops
from gstaichi.lang._fast_caching import src_hasher
from gstaichi.lang._template_mapper import TemplateMapper
from gstaichi.lang._wrap_inspect import FunctionSourceInfo, get_source_info_and_src
from gstaichi.lang.any_array import AnyArray
from gstaichi.lang.ast import (
    ASTTransformerContext,
    KernelSimplicityASTChecker,
    transform_tree,
)
from gstaichi.lang.ast.ast_transformer_utils import ReturnStatus
from gstaichi.lang.exception import (
    GsTaichiCompilationError,
    GsTaichiRuntimeError,
    GsTaichiRuntimeTypeError,
    GsTaichiSyntaxError,
    GsTaichiTypeError,
    handle_exception_from_cpp,
)
from gstaichi.lang.expr import Expr
from gstaichi.lang.kernel_arguments import ArgMetadata
from gstaichi.lang.matrix import MatrixType
from gstaichi.lang.shell import _shell_pop_print
from gstaichi.lang.struct import StructType
from gstaichi.lang.util import cook_dtype, has_pytorch
from gstaichi.types import (
    ndarray_type,
    primitive_types,
    sparse_matrix_builder,
    template,
    texture_type,
)
from gstaichi.types.compound_types import CompoundType
from gstaichi.types.enums import AutodiffMode, Layout
from gstaichi.types.utils import is_signed

CompiledKernelKeyType = tuple[Callable, int, AutodiffMode]


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
        self._primal: Kernel | None = None
        self._adjoint: Kernel | None = None
        self.grad: Kernel | None = None
        self._is_staticmethod: bool = False
        self.is_pure = False
        functools.update_wrapper(self, fn)

    def __call__(self, *args, **kwargs):
        return self.wrapper.__call__(*args, **kwargs)

    def __get__(self, instance, owner):
        if instance is None:
            return self
        return BoundGsTaichiCallable(instance, self)


class BoundGsTaichiCallable:
    def __init__(self, instance: Any, gstaichi_callable: "GsTaichiCallable"):
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
        if k in ("wrapper", "instance", "gstaichi_callable"):
            object.__setattr__(self, k, v)
        else:
            setattr(self.gstaichi_callable, k, v)


def func(fn: Callable, is_real_function: bool = False) -> GsTaichiCallable:
    """Marks a function as callable in GsTaichi-scope.

    This decorator transforms a Python function into a GsTaichi one. GsTaichi
    will JIT compile it into native instructions.

    Args:
        fn (Callable): The Python function to be decorated
        is_real_function (bool): Whether the function is a real function

    Returns:
        Callable: The decorated function

    Example::

        >>> @ti.func
        >>> def foo(x):
        >>>     return x + 2
        >>>
        >>> @ti.kernel
        >>> def run():
        >>>     print(foo(40))  # 42
    """
    is_classfunc = _inside_class(level_of_class_stackframe=3 + is_real_function)

    fun = Func(fn, _classfunc=is_classfunc, is_real_function=is_real_function)
    gstaichi_callable = GsTaichiCallable(fn, fun)
    gstaichi_callable._is_gstaichi_function = True
    gstaichi_callable._is_real_function = is_real_function
    return gstaichi_callable


def real_func(fn: Callable) -> GsTaichiCallable:
    return func(fn, is_real_function=True)


def pyfunc(fn: Callable) -> GsTaichiCallable:
    """Marks a function as callable in both GsTaichi and Python scopes.

    When called inside the GsTaichi scope, GsTaichi will JIT compile it into
    native instructions. Otherwise it will be invoked directly as a
    Python function.

    See also :func:`~gstaichi.lang.kernel_impl.func`.

    Args:
        fn (Callable): The Python function to be decorated

    Returns:
        Callable: The decorated function
    """
    is_classfunc = _inside_class(level_of_class_stackframe=3)
    fun = Func(fn, _classfunc=is_classfunc, _pyfunc=True)
    gstaichi_callable = GsTaichiCallable(fn, fun)
    gstaichi_callable._is_gstaichi_function = True
    gstaichi_callable._is_real_function = False
    return gstaichi_callable


def _populate_global_vars_for_templates(
    template_slot_locations: list[int],
    argument_metas: list[ArgMetadata],
    global_vars: dict[str, Any],
    fn: Callable,
    py_args: tuple[Any, ...],
):
    """
    Inject template parameters into globals

    Globals are being abused to store the python objects associated
    with templates. We continue this approach, and in addition this function
    handles injecting expanded python variables from dataclasses.
    """
    for i in template_slot_locations:
        template_var_name = argument_metas[i].name
        global_vars[template_var_name] = py_args[i]
    parameters = inspect.signature(fn).parameters
    for i, (parameter_name, parameter) in enumerate(parameters.items()):
        if dataclasses.is_dataclass(parameter.annotation):
            _kernel_impl_dataclass.populate_global_vars_from_dataclass(
                parameter_name,
                parameter.annotation,
                py_args[i],
                global_vars=global_vars,
            )


def _get_tree_and_ctx(
    self: "Func | Kernel",
    args: tuple[Any, ...],
    excluded_parameters=(),
    is_kernel: bool = True,
    arg_features=None,
    ast_builder: "ASTBuilder | None" = None,
    is_real_function: bool = False,
    current_kernel: "Kernel | None" = None,
) -> tuple[ast.Module, ASTTransformerContext]:
    function_source_info, src = get_source_info_and_src(self.func)
    src = [textwrap.fill(line, tabsize=4, width=9999) for line in src]
    tree = ast.parse(textwrap.dedent("\n".join(src)))

    func_body = tree.body[0]
    func_body.decorator_list = []  # type: ignore , kick that can down the road...

    global_vars = _get_global_vars(self.func)

    if is_kernel or is_real_function:
        _populate_global_vars_for_templates(
            template_slot_locations=self.template_slot_locations,
            argument_metas=self.arg_metas,
            global_vars=global_vars,
            fn=self.func,
            py_args=args,
        )

    if current_kernel is not None:  # Kernel
        current_kernel.kernel_function_info = function_source_info
    if current_kernel is None:
        current_kernel = impl.get_runtime()._current_kernel
    assert current_kernel is not None
    current_kernel.visited_functions.add(function_source_info)

    return tree, ASTTransformerContext(
        excluded_parameters=excluded_parameters,
        is_kernel=is_kernel,
        func=self,
        arg_features=arg_features,
        global_vars=global_vars,
        argument_data=args,
        src=src,
        start_lineno=function_source_info.start_lineno,
        end_lineno=function_source_info.end_lineno,
        file=function_source_info.filepath,
        ast_builder=ast_builder,
        is_real_function=is_real_function,
    )


def _process_args(self: "Func | Kernel", is_func: bool, args: tuple[Any, ...], kwargs) -> tuple[Any, ...]:
    if is_func:
        self.arg_metas = _kernel_impl_dataclass.expand_func_arguments(self.arg_metas)

    fused_args: list[Any] = [arg_meta.default for arg_meta in self.arg_metas]
    len_args = len(args)

    if len_args > len(fused_args):
        arg_str = ", ".join(map(str, args))
        expected_str = ", ".join(f"{arg.name} : {arg.annotation}" for arg in self.arg_metas)
        msg = f"Too many arguments. Expected ({expected_str}), got ({arg_str})."
        raise GsTaichiSyntaxError(msg)

    for i, arg in enumerate(args):
        fused_args[i] = arg

    for key, value in kwargs.items():
        for i, arg in enumerate(self.arg_metas):
            if key == arg.name:
                if i < len_args:
                    raise GsTaichiSyntaxError(f"Multiple values for argument '{key}'.")
                fused_args[i] = value
                break
        else:
            raise GsTaichiSyntaxError(f"Unexpected argument '{key}'.")

    for i, arg in enumerate(fused_args):
        if arg is inspect.Parameter.empty:
            if self.arg_metas[i].annotation is inspect._empty:
                raise GsTaichiSyntaxError(f"Parameter `{self.arg_metas[i].name}` missing.")
            else:
                raise GsTaichiSyntaxError(
                    f"Parameter `{self.arg_metas[i].name} : {self.arg_metas[i].annotation}` missing."
                )

    return tuple(fused_args)


class Func:
    function_counter = 0

    def __init__(self, _func: Callable, _classfunc=False, _pyfunc=False, is_real_function=False) -> None:
        self.func = _func
        self.func_id = Func.function_counter
        Func.function_counter += 1
        self.compiled = {}
        self.classfunc = _classfunc
        self.pyfunc = _pyfunc
        self.is_real_function = is_real_function
        self.arg_metas: list[ArgMetadata] = []
        self.orig_arguments: list[ArgMetadata] = []
        self.return_type: tuple[Type, ...] | None = None
        self.extract_arguments()
        self.template_slot_locations: list[int] = []
        for i, arg in enumerate(self.arg_metas):
            if arg.annotation == template or isinstance(arg.annotation, template):
                self.template_slot_locations.append(i)
        self.mapper = TemplateMapper(self.arg_metas, self.template_slot_locations)
        self.gstaichi_functions = {}  # The |Function| class in C++
        self.has_print = False

    def __call__(self: "Func", *args, **kwargs) -> Any:
        args = _process_args(self, is_func=True, args=args, kwargs=kwargs)

        if not impl.inside_kernel():
            if not self.pyfunc:
                raise GsTaichiSyntaxError("GsTaichi functions cannot be called from Python-scope.")
            return self.func(*args)

        current_kernel = impl.get_runtime().current_kernel
        if self.is_real_function:
            if current_kernel.autodiff_mode != AutodiffMode.NONE:
                raise GsTaichiSyntaxError("Real function in gradient kernels unsupported.")
            instance_id, arg_features = self.mapper.lookup(args)
            key = _ti_core.FunctionKey(self.func.__name__, self.func_id, instance_id)
            if key.instance_id not in self.compiled:
                self.do_compile(key=key, args=args, arg_features=arg_features)
            return self.func_call_rvalue(key=key, args=args)
        tree, ctx = _get_tree_and_ctx(
            self,
            is_kernel=False,
            args=args,
            ast_builder=current_kernel.ast_builder(),
            is_real_function=self.is_real_function,
        )

        struct_locals = _kernel_impl_dataclass.extract_struct_locals_from_context(ctx)

        tree = _kernel_impl_dataclass.unpack_ast_struct_expressions(tree, struct_locals=struct_locals)
        ret = transform_tree(tree, ctx)
        if not self.is_real_function:
            if self.return_type and ctx.returned != ReturnStatus.ReturnedValue:
                raise GsTaichiSyntaxError("Function has a return type but does not have a return statement")
        return ret

    def func_call_rvalue(self, key: FunctionKey, args: tuple[Any, ...]) -> Any:
        # Skip the template args, e.g., |self|
        assert self.is_real_function
        non_template_args = []
        dbg_info = _ti_core.DebugInfo(impl.get_runtime().get_current_src_info())
        for i, kernel_arg in enumerate(self.arg_metas):
            anno = kernel_arg.annotation
            if not isinstance(anno, template):
                if id(anno) in primitive_types.type_ids:
                    non_template_args.append(ops.cast(args[i], anno))
                elif isinstance(anno, primitive_types.RefType):
                    non_template_args.append(_ti_core.make_reference(args[i].ptr, dbg_info))
                elif isinstance(anno, ndarray_type.NdarrayType):
                    if not isinstance(args[i], AnyArray):
                        raise GsTaichiTypeError(
                            f"Expected ndarray in the kernel argument for argument {kernel_arg.name}, got {args[i]}"
                        )
                    non_template_args += _ti_core.get_external_tensor_real_func_args(args[i].ptr, dbg_info)
                else:
                    non_template_args.append(args[i])
        non_template_args = impl.make_expr_group(non_template_args)
        compiling_callable = impl.get_runtime().compiling_callable
        assert compiling_callable is not None
        func_call = compiling_callable.ast_builder().insert_func_call(
            self.gstaichi_functions[key.instance_id], non_template_args, dbg_info
        )
        if self.return_type is None:
            return None
        func_call = Expr(func_call)
        ret = []

        for i, return_type in enumerate(self.return_type):
            if id(return_type) in primitive_types.type_ids:
                ret.append(
                    Expr(
                        _ti_core.make_get_element_expr(
                            func_call.ptr, (i,), _ti_core.DebugInfo(impl.get_runtime().get_current_src_info())
                        )
                    )
                )
            elif isinstance(return_type, (StructType, MatrixType)):
                ret.append(return_type.from_gstaichi_object(func_call, (i,)))
            else:
                raise GsTaichiTypeError(f"Unsupported return type for return value {i}: {return_type}")
        if len(ret) == 1:
            return ret[0]
        return tuple(ret)

    def do_compile(self, key: FunctionKey, args: tuple[Any, ...], arg_features: tuple[Any, ...]) -> None:
        tree, ctx = _get_tree_and_ctx(
            self, is_kernel=False, args=args, arg_features=arg_features, is_real_function=self.is_real_function
        )
        fn = impl.get_runtime().prog.create_function(key)

        def func_body():
            old_callable = impl.get_runtime().compiling_callable
            impl.get_runtime()._compiling_callable = fn
            ctx.ast_builder = fn.ast_builder()
            transform_tree(tree, ctx)
            impl.get_runtime()._compiling_callable = old_callable

        self.gstaichi_functions[key.instance_id] = fn
        self.compiled[key.instance_id] = func_body
        self.gstaichi_functions[key.instance_id].set_function_body(func_body)

    def extract_arguments(self) -> None:
        sig = inspect.signature(self.func)
        if sig.return_annotation not in (inspect.Signature.empty, None):
            self.return_type = sig.return_annotation
            if (
                isinstance(self.return_type, (types.GenericAlias, typing._GenericAlias))  # type: ignore
                and self.return_type.__origin__ is tuple  # type: ignore
            ):
                self.return_type = self.return_type.__args__  # type: ignore
            if self.return_type is None:
                return
            if not isinstance(self.return_type, (list, tuple)):
                self.return_type = (self.return_type,)
            for i, return_type in enumerate(self.return_type):
                if return_type is Ellipsis:
                    raise GsTaichiSyntaxError("Ellipsis is not supported in return type annotations")
        params = sig.parameters
        arg_names = params.keys()
        for i, arg_name in enumerate(arg_names):
            param = params[arg_name]
            if param.kind == inspect.Parameter.VAR_KEYWORD:
                raise GsTaichiSyntaxError(
                    "GsTaichi functions do not support variable keyword parameters (i.e., **kwargs)"
                )
            if param.kind == inspect.Parameter.VAR_POSITIONAL:
                raise GsTaichiSyntaxError(
                    "GsTaichi functions do not support variable positional parameters (i.e., *args)"
                )
            if param.kind == inspect.Parameter.KEYWORD_ONLY:
                raise GsTaichiSyntaxError("GsTaichi functions do not support keyword parameters")
            if param.kind != inspect.Parameter.POSITIONAL_OR_KEYWORD:
                raise GsTaichiSyntaxError('GsTaichi functions only support "positional or keyword" parameters')
            annotation = param.annotation
            if annotation is inspect.Parameter.empty:
                if i == 0 and self.classfunc:
                    annotation = template()
                # TODO: pyfunc also need type annotation check when real function is enabled,
                #       but that has to happen at runtime when we know which scope it's called from.
                elif not self.pyfunc and self.is_real_function:
                    raise GsTaichiSyntaxError(
                        f"GsTaichi function `{self.func.__name__}` parameter `{arg_name}` must be type annotated"
                    )
            else:
                if isinstance(annotation, ndarray_type.NdarrayType):
                    pass
                elif isinstance(annotation, MatrixType):
                    pass
                elif isinstance(annotation, StructType):
                    pass
                elif id(annotation) in primitive_types.type_ids:
                    pass
                elif type(annotation) == gstaichi.types.annotations.Template:
                    pass
                elif isinstance(annotation, template) or annotation == gstaichi.types.annotations.Template:
                    pass
                elif isinstance(annotation, primitive_types.RefType):
                    pass
                elif isinstance(annotation, type) and dataclasses.is_dataclass(annotation):
                    pass
                else:
                    raise GsTaichiSyntaxError(
                        f"Invalid type annotation (argument {i}) of GsTaichi function: {annotation}"
                    )
            self.arg_metas.append(ArgMetadata(annotation, param.name, param.default))
            self.orig_arguments.append(ArgMetadata(annotation, param.name, param.default))


def _get_global_vars(_func: Callable) -> dict[str, Any]:
    # Discussions: https://github.com/taichi-dev/gstaichi/issues/282
    global_vars = _func.__globals__.copy()

    freevar_names = _func.__code__.co_freevars
    closure = _func.__closure__
    if closure:
        freevar_values = list(map(lambda x: x.cell_contents, closure))
        for name, value in zip(freevar_names, freevar_values):
            global_vars[name] = value

    return global_vars


@dataclasses.dataclass
class SrcLlCacheObservations:
    cache_key_generated: bool = False
    cache_validated: bool = False
    cache_loaded: bool = False
    cache_stored: bool = False


@dataclasses.dataclass
class FeLlCacheObservations:
    cache_hit: bool = False


class Kernel:
    counter = 0

    def __init__(self, _func: Callable, autodiff_mode: AutodiffMode, _classkernel=False) -> None:
        self.func = _func
        self.kernel_counter = Kernel.counter
        Kernel.counter += 1
        assert autodiff_mode in (
            AutodiffMode.NONE,
            AutodiffMode.VALIDATION,
            AutodiffMode.FORWARD,
            AutodiffMode.REVERSE,
        )
        self.autodiff_mode = autodiff_mode
        self.grad: "Kernel | None" = None
        self.arg_metas: list[ArgMetadata] = []
        self.return_type = None
        self.classkernel = _classkernel
        self.extract_arguments()
        self.template_slot_locations = []
        for i, arg in enumerate(self.arg_metas):
            if arg.annotation == template or isinstance(arg.annotation, template):
                self.template_slot_locations.append(i)
        self.mapper = TemplateMapper(self.arg_metas, self.template_slot_locations)
        impl.get_runtime().kernels.append(self)
        self.reset()
        self.kernel_cpp = None
        # A materialized kernel is a KernelCxx object which may or may not have
        # been compiled. It generally has been converted at least as far as AST
        # and front-end IR, but not necessarily any further.
        self.materialized_kernels: dict[CompiledKernelKeyType, KernelCxx] = {}
        self.has_print = False
        self.gstaichi_callable: GsTaichiCallable | None = None
        self.visited_functions: set[FunctionSourceInfo] = set()
        self.kernel_function_info: FunctionSourceInfo | None = None
        self.compiled_kernel_data_by_key: dict[CompiledKernelKeyType, CompiledKernelData] = {}

        self.src_ll_cache_observations: SrcLlCacheObservations = SrcLlCacheObservations()
        self.fe_ll_cache_observations: FeLlCacheObservations = FeLlCacheObservations()

    def ast_builder(self) -> ASTBuilder:
        assert self.kernel_cpp is not None
        return self.kernel_cpp.ast_builder()

    def reset(self) -> None:
        self.runtime = impl.get_runtime()
        self.materialized_kernels = {}

    def extract_arguments(self) -> None:
        sig = inspect.signature(self.func)
        if sig.return_annotation not in (inspect._empty, None):
            self.return_type = sig.return_annotation
            if (
                isinstance(self.return_type, (types.GenericAlias, typing._GenericAlias))  # type: ignore
                and self.return_type.__origin__ is tuple
            ):
                self.return_type = self.return_type.__args__
            if not isinstance(self.return_type, (list, tuple)):
                self.return_type = (self.return_type,)
            for return_type in self.return_type:
                if return_type is Ellipsis:
                    raise GsTaichiSyntaxError("Ellipsis is not supported in return type annotations")
        params = dict(sig.parameters)
        arg_names = params.keys()
        for i, arg_name in enumerate(arg_names):
            param = params[arg_name]
            if param.kind == inspect.Parameter.VAR_KEYWORD:
                raise GsTaichiSyntaxError(
                    "GsTaichi kernels do not support variable keyword parameters (i.e., **kwargs)"
                )
            if param.kind == inspect.Parameter.VAR_POSITIONAL:
                raise GsTaichiSyntaxError(
                    "GsTaichi kernels do not support variable positional parameters (i.e., *args)"
                )
            if param.default is not inspect.Parameter.empty:
                raise GsTaichiSyntaxError("GsTaichi kernels do not support default values for arguments")
            if param.kind == inspect.Parameter.KEYWORD_ONLY:
                raise GsTaichiSyntaxError("GsTaichi kernels do not support keyword parameters")
            if param.kind != inspect.Parameter.POSITIONAL_OR_KEYWORD:
                raise GsTaichiSyntaxError('GsTaichi kernels only support "positional or keyword" parameters')
            annotation = param.annotation
            if param.annotation is inspect.Parameter.empty:
                if i == 0 and self.classkernel:  # The |self| parameter
                    annotation = template()
                else:
                    raise GsTaichiSyntaxError("GsTaichi kernels parameters must be type annotated")
            else:
                if isinstance(
                    annotation,
                    (
                        template,
                        ndarray_type.NdarrayType,
                        texture_type.TextureType,
                        texture_type.RWTextureType,
                    ),
                ):
                    pass
                elif annotation is ndarray_type.NdarrayType:
                    # convert from ti.types.NDArray into ti.types.NDArray()
                    annotation = annotation()
                elif id(annotation) in primitive_types.type_ids:
                    pass
                elif isinstance(annotation, sparse_matrix_builder):
                    pass
                elif isinstance(annotation, MatrixType):
                    pass
                elif isinstance(annotation, StructType):
                    pass
                elif annotation == template:
                    pass
                elif isinstance(annotation, type) and dataclasses.is_dataclass(annotation):
                    pass
                else:
                    raise GsTaichiSyntaxError(f"Invalid type annotation (argument {i}) of Taichi kernel: {annotation}")
            self.arg_metas.append(ArgMetadata(annotation, param.name, param.default))

    def materialize(self, key: CompiledKernelKeyType | None, args: tuple[Any, ...], arg_features=None):
        if key is None:
            key = (self.func, 0, self.autodiff_mode)
        self.runtime.materialize()
        self.fast_checksum = None

        if key in self.materialized_kernels:
            return

        if self.runtime.src_ll_cache and self.gstaichi_callable and self.gstaichi_callable.is_pure:
            kernel_source_info, _src = get_source_info_and_src(self.func)
            self.fast_checksum = src_hasher.create_cache_key(kernel_source_info, args)
            if self.fast_checksum:
                self.src_ll_cache_observations.cache_key_generated = True
            if self.fast_checksum and src_hasher.validate_cache_key(self.fast_checksum):
                self.src_ll_cache_observations.cache_validated = True
                prog = impl.get_runtime().prog
                self.compiled_kernel_data_by_key[key] = prog.load_fast_cache(
                    self.fast_checksum,
                    self.func.__name__,
                    prog.config(),
                    prog.get_device_caps(),
                )
                if self.compiled_kernel_data_by_key[key]:
                    self.src_ll_cache_observations.cache_loaded = True
        elif self.gstaichi_callable and not self.gstaichi_callable.is_pure and self.runtime.print_non_pure:
            # The bit in caps should not be modified without updating corresponding test
            # freetext can be freely modified.
            # As for why we are using `print` rather than eg logger.info, it is because
            # this is only printed when ti.init(print_non_pure=..) is True. And it is
            # confusing to set that to True, and see nothing printed.
            print(f"[NOT_PURE] Debug information: not pure: {self.func.__name__}")

        kernel_name = f"{self.func.__name__}_c{self.kernel_counter}_{key[1]}"
        _logging.trace(f"Materializing kernel {kernel_name} in {self.autodiff_mode}...")

        tree, ctx = _get_tree_and_ctx(
            self,
            args=args,
            excluded_parameters=self.template_slot_locations,
            arg_features=arg_features,
            current_kernel=self,
        )

        if self.autodiff_mode != AutodiffMode.NONE:
            KernelSimplicityASTChecker(self.func).visit(tree)

        # Do not change the name of 'gstaichi_ast_generator'
        # The warning system needs this identifier to remove unnecessary messages
        def gstaichi_ast_generator(kernel_cxx: KernelCxx):
            nonlocal tree
            if self.runtime.inside_kernel:
                raise GsTaichiSyntaxError(
                    "Kernels cannot call other kernels. I.e., nested kernels are not allowed. "
                    "Please check if you have direct/indirect invocation of kernels within kernels. "
                    "Note that some methods provided by the GsTaichi standard library may invoke kernels, "
                    "and please move their invocations to Python-scope."
                )
            self.kernel_cpp = kernel_cxx
            self.runtime.inside_kernel = True
            self.runtime._current_kernel = self
            assert self.runtime._compiling_callable is None
            self.runtime._compiling_callable = kernel_cxx
            try:
                ctx.ast_builder = kernel_cxx.ast_builder()

                def ast_to_dict(node: ast.AST | list | primitive_types._python_primitive_types):
                    if isinstance(node, ast.AST):
                        fields = {k: ast_to_dict(v) for k, v in ast.iter_fields(node)}
                        return {
                            "type": node.__class__.__name__,
                            "fields": fields,
                            "lineno": getattr(node, "lineno", None),
                            "col_offset": getattr(node, "col_offset", None),
                        }
                    if isinstance(node, list):
                        return [ast_to_dict(x) for x in node]
                    return node  # Basic types (str, int, None, etc.)

                if os.environ.get("TI_DUMP_AST", "") == "1":
                    target_dir = pathlib.Path("/tmp/ast")
                    target_dir.mkdir(parents=True, exist_ok=True)

                    start = time.time()
                    ast_str = ast.dump(tree, indent=2)
                    output_file = target_dir / f"{kernel_name}_ast.txt"
                    output_file.write_text(ast_str)
                    elapsed_txt = time.time() - start

                    start = time.time()
                    json_str = json.dumps(ast_to_dict(tree), indent=2)
                    output_file = target_dir / f"{kernel_name}_ast.json"
                    output_file.write_text(json_str)
                    elapsed_json = time.time() - start

                    output_file = target_dir / f"{kernel_name}_gen_time.json"
                    output_file.write_text(
                        json.dumps({"elapsed_txt": elapsed_txt, "elapsed_json": elapsed_json}, indent=2)
                    )
                struct_locals = _kernel_impl_dataclass.extract_struct_locals_from_context(ctx)
                tree = _kernel_impl_dataclass.unpack_ast_struct_expressions(tree, struct_locals=struct_locals)
                ctx.only_parse_function_def = self.compiled_kernel_data_by_key.get(key) is not None
                transform_tree(tree, ctx)
                if not ctx.is_real_function:
                    if self.return_type and ctx.returned != ReturnStatus.ReturnedValue:
                        raise GsTaichiSyntaxError("Kernel has a return type but does not have a return statement")
            finally:
                self.runtime.inside_kernel = False
                self.runtime._current_kernel = None
                self.runtime._compiling_callable = None

        gstaichi_kernel = impl.get_runtime().prog.create_kernel(gstaichi_ast_generator, kernel_name, self.autodiff_mode)
        assert key not in self.materialized_kernels
        self.materialized_kernels[key] = gstaichi_kernel

    def launch_kernel(self, t_kernel: KernelCxx, compiled_kernel_data: CompiledKernelData | None, *args) -> Any:
        assert len(args) == len(self.arg_metas), f"{len(self.arg_metas)} arguments needed but {len(args)} provided"

        tmps = []
        callbacks = []

        actual_argument_slot = 0
        launch_ctx = t_kernel.make_launch_context()
        max_arg_num = 512
        exceed_max_arg_num = False

        def set_arg_ndarray(indices: tuple[int, ...], v: gstaichi.lang._ndarray.Ndarray) -> None:
            v_primal = v.arr
            v_grad = v.grad.arr if v.grad else None
            if v_grad is None:
                launch_ctx.set_arg_ndarray(indices, v_primal)  # type: ignore , solvable probably, just not today
            else:
                launch_ctx.set_arg_ndarray_with_grad(indices, v_primal, v_grad)  # type: ignore

        def set_arg_texture(indices: tuple[int, ...], v: gstaichi.lang._texture.Texture) -> None:
            launch_ctx.set_arg_texture(indices, v.tex)

        def set_arg_rw_texture(indices: tuple[int, ...], v: gstaichi.lang._texture.Texture) -> None:
            launch_ctx.set_arg_rw_texture(indices, v.tex)

        def set_arg_ext_array(indices: tuple[int, ...], v: Any, needed: ndarray_type.NdarrayType) -> None:
            # v is things like torch Tensor and numpy array
            # Not adding type for this, since adds additional dependencies
            #
            # Element shapes are already specialized in GsTaichi codegen.
            # The shape information for element dims are no longer needed.
            # Therefore we strip the element shapes from the shape vector,
            # so that it only holds "real" array shapes.
            is_soa = needed.layout == Layout.SOA
            array_shape = v.shape
            if functools.reduce(operator.mul, array_shape, 1) > np.iinfo(np.int32).max:
                warnings.warn("Ndarray index might be out of int32 boundary but int64 indexing is not supported yet.")
            if needed.dtype is None or id(needed.dtype) in primitive_types.type_ids:
                element_dim = 0
            else:
                element_dim = needed.dtype.ndim
                array_shape = v.shape[element_dim:] if is_soa else v.shape[:-element_dim]
            if isinstance(v, np.ndarray):
                # numpy
                if v.flags.c_contiguous:
                    launch_ctx.set_arg_external_array_with_shape(indices, int(v.ctypes.data), v.nbytes, array_shape, 0)
                elif v.flags.f_contiguous:
                    # TODO: A better way that avoids copying is saving strides info.
                    tmp = np.ascontiguousarray(v)
                    # Purpose: DO NOT GC |tmp|!
                    tmps.append(tmp)

                    def callback(original, updated):
                        np.copyto(original, np.asfortranarray(updated))

                    callbacks.append(functools.partial(callback, v, tmp))
                    launch_ctx.set_arg_external_array_with_shape(
                        indices, int(tmp.ctypes.data), tmp.nbytes, array_shape, 0
                    )
                else:
                    raise ValueError(
                        "Non contiguous numpy arrays are not supported, please call np.ascontiguousarray(arr) "
                        "before passing it into gstaichi kernel."
                    )
            elif has_pytorch():
                import torch  # pylint: disable=C0415

                if isinstance(v, torch.Tensor):
                    if not v.is_contiguous():
                        raise ValueError(
                            "Non contiguous tensors are not supported, please call tensor.contiguous() before "
                            "passing it into gstaichi kernel."
                        )
                    gstaichi_arch = self.runtime.prog.config().arch

                    def get_call_back(u, v):
                        def call_back():
                            u.copy_(v)

                        return call_back

                    # FIXME: only allocate when launching grad kernel
                    if v.requires_grad and v.grad is None:
                        v.grad = torch.zeros_like(v)

                    if v.requires_grad:
                        if not isinstance(v.grad, torch.Tensor):
                            raise ValueError(
                                f"Expecting torch.Tensor for gradient tensor, but getting {v.grad.__class__.__name__} instead"
                            )
                        if not v.grad.is_contiguous():
                            raise ValueError(
                                "Non contiguous gradient tensors are not supported, please call tensor.grad.contiguous() before passing it into gstaichi kernel."
                            )

                    tmp = v
                    if (str(v.device) != "cpu") and not (
                        str(v.device).startswith("cuda") and gstaichi_arch == _ti_core.Arch.cuda
                    ):
                        # Getting a torch CUDA tensor on GsTaichi non-cuda arch:
                        # We just replace it with a CPU tensor and by the end of kernel execution we'll use the
                        # callback to copy the values back to the original CUDA tensor.
                        host_v = v.to(device="cpu", copy=True)
                        tmp = host_v
                        callbacks.append(get_call_back(v, host_v))

                    launch_ctx.set_arg_external_array_with_shape(
                        indices,
                        int(tmp.data_ptr()),
                        tmp.element_size() * tmp.nelement(),
                        array_shape,
                        int(v.grad.data_ptr()) if v.grad is not None else 0,
                    )
                else:
                    raise GsTaichiRuntimeTypeError(
                        f"Argument of type {type(v)} cannot be converted into required type {needed}"
                    )
            else:
                raise GsTaichiRuntimeTypeError(f"Argument {needed} cannot be converted into required type {v}")

        def set_arg_matrix(indices: tuple[int, ...], v, needed) -> None:
            def cast_float(x: float | np.floating | np.integer | int) -> float:
                if not isinstance(x, (int, float, np.integer, np.floating)):
                    raise GsTaichiRuntimeTypeError(
                        f"Argument {needed.dtype} cannot be converted into required type {type(x)}"
                    )
                return float(x)

            def cast_int(x: int | np.integer) -> int:
                if not isinstance(x, (int, np.integer)):
                    raise GsTaichiRuntimeTypeError(
                        f"Argument {needed.dtype} cannot be converted into required type {type(x)}"
                    )
                return int(x)

            cast_func = None
            if needed.dtype in primitive_types.real_types:
                cast_func = cast_float
            elif needed.dtype in primitive_types.integer_types:
                cast_func = cast_int
            else:
                raise ValueError(f"Matrix dtype {needed.dtype} is not integer type or real type.")

            if needed.ndim == 2:
                v = [cast_func(v[i, j]) for i in range(needed.n) for j in range(needed.m)]
            else:
                v = [cast_func(v[i]) for i in range(needed.n)]
            v = needed(*v)
            needed.set_kernel_struct_args(v, launch_ctx, indices)

        def set_arg_sparse_matrix_builder(indices: tuple[int, ...], v) -> None:
            # Pass only the base pointer of the ti.types.sparse_matrix_builder() argument
            launch_ctx.set_arg_uint(indices, v._get_ndarray_addr())

        set_later_list = []

        def recursive_set_args(needed_arg_type: Type, provided_arg_type: Type, v: Any, indices: tuple[int, ...]) -> int:
            """
            Returns the number of kernel args set
            e.g. templates don't set kernel args, so returns 0
            a single ndarray is 1 kernel arg, so returns 1
            a struct of 3 ndarrays would set 3 kernel args, so return 3
            note: len(indices) > 1 only happens with argpack (which we are removing support for)
            """
            nonlocal actual_argument_slot, exceed_max_arg_num, set_later_list
            if actual_argument_slot >= max_arg_num:
                exceed_max_arg_num = True
                return 0
            actual_argument_slot += 1
            # Note: do not use sth like "needed == f32". That would be slow.
            if id(needed_arg_type) in primitive_types.real_type_ids:
                if not isinstance(v, (float, int, np.floating, np.integer)):
                    raise GsTaichiRuntimeTypeError.get(indices, needed_arg_type.to_string(), provided_arg_type)
                launch_ctx.set_arg_float(indices, float(v))
                return 1
            if id(needed_arg_type) in primitive_types.integer_type_ids:
                if not isinstance(v, (int, np.integer)):
                    raise GsTaichiRuntimeTypeError.get(indices, needed_arg_type.to_string(), provided_arg_type)
                if is_signed(cook_dtype(needed_arg_type)):
                    launch_ctx.set_arg_int(indices, int(v))
                else:
                    launch_ctx.set_arg_uint(indices, int(v))
                return 1
            if isinstance(needed_arg_type, sparse_matrix_builder):
                set_arg_sparse_matrix_builder(indices, v)
                return 1
            if dataclasses.is_dataclass(needed_arg_type):
                assert provided_arg_type == needed_arg_type
                idx = 0
                for j, field in enumerate(dataclasses.fields(needed_arg_type)):
                    assert not isinstance(field.type, str)
                    field_value = getattr(v, field.name)
                    idx += recursive_set_args(field.type, field.type, field_value, (indices[0] + idx,))
                return idx
            if isinstance(needed_arg_type, ndarray_type.NdarrayType) and isinstance(v, gstaichi.lang._ndarray.Ndarray):
                set_arg_ndarray(indices, v)
                return 1
            if isinstance(needed_arg_type, texture_type.TextureType) and isinstance(v, gstaichi.lang._texture.Texture):
                set_arg_texture(indices, v)
                return 1
            if isinstance(needed_arg_type, texture_type.RWTextureType) and isinstance(
                v, gstaichi.lang._texture.Texture
            ):
                set_arg_rw_texture(indices, v)
                return 1
            if isinstance(needed_arg_type, ndarray_type.NdarrayType):
                set_arg_ext_array(indices, v, needed_arg_type)
                return 1
            if isinstance(needed_arg_type, MatrixType):
                set_arg_matrix(indices, v, needed_arg_type)
                return 1
            if isinstance(needed_arg_type, StructType):
                # Unclear how to make the following pass typing checks
                # StructType implements __instancecheck__, which should be a classmethod, but
                # is currently an instance method
                # TODO: look into this more deeply at some point
                if not isinstance(v, needed_arg_type):  # type: ignore
                    raise GsTaichiRuntimeTypeError(
                        f"Argument {provided_arg_type} cannot be converted into required type {needed_arg_type}"
                    )
                needed_arg_type.set_kernel_struct_args(v, launch_ctx, indices)
                return 1
            if needed_arg_type == template or isinstance(needed_arg_type, template):
                return 0
            raise ValueError(f"Argument type mismatch. Expecting {needed_arg_type}, got {type(v)}.")

        template_num = 0
        i_out = 0
        for i_in, val in enumerate(args):
            needed_ = self.arg_metas[i_in].annotation
            if needed_ == template or isinstance(needed_, template):
                template_num += 1
                i_out += 1
                continue
            i_out += recursive_set_args(needed_, type(val), val, (i_out - template_num,))

        for i, (set_arg_func, params) in enumerate(set_later_list):
            set_arg_func((len(args) - template_num + i,), *params)

        if exceed_max_arg_num:
            raise GsTaichiRuntimeError(
                f"The number of elements in kernel arguments is too big! Do not exceed {max_arg_num} on {_ti_core.arch_name(impl.current_cfg().arch)} backend."
            )

        try:
            prog = impl.get_runtime().prog
            if not compiled_kernel_data:
                compile_result: CompileResult = prog.compile_kernel(prog.config(), prog.get_device_caps(), t_kernel)
                compiled_kernel_data = compile_result.compiled_kernel_data
                if compile_result.cache_hit:
                    self.fe_ll_cache_observations.cache_hit = True
                if self.fast_checksum:
                    src_hasher.store(self.fast_checksum, self.visited_functions)
                    prog.store_fast_cache(
                        self.fast_checksum,
                        self.kernel_cpp,
                        prog.config(),
                        prog.get_device_caps(),
                        compiled_kernel_data,
                    )
                    self.src_ll_cache_observations.cache_stored = True
            prog.launch_kernel(compiled_kernel_data, launch_ctx)
        except Exception as e:
            e = handle_exception_from_cpp(e)
            if impl.get_runtime().print_full_traceback:
                raise e
            raise e from None

        ret = None
        ret_dt = self.return_type
        has_ret = ret_dt is not None

        if has_ret or self.has_print:
            runtime_ops.sync()

        if has_ret:
            ret = []
            for i, ret_type in enumerate(ret_dt):
                ret.append(self.construct_kernel_ret(launch_ctx, ret_type, (i,)))
            if len(ret_dt) == 1:
                ret = ret[0]
        if callbacks:
            for c in callbacks:
                c()

        return ret

    def construct_kernel_ret(self, launch_ctx: KernelLaunchContext, ret_type: Any, index: tuple[int, ...] = ()):
        if isinstance(ret_type, CompoundType):
            return ret_type.from_kernel_struct_ret(launch_ctx, index)
        if ret_type in primitive_types.integer_types:
            if is_signed(cook_dtype(ret_type)):
                return launch_ctx.get_struct_ret_int(index)
            return launch_ctx.get_struct_ret_uint(index)
        if ret_type in primitive_types.real_types:
            return launch_ctx.get_struct_ret_float(index)
        raise GsTaichiRuntimeTypeError(f"Invalid return type on index={index}")

    def ensure_compiled(self, *args: tuple[Any, ...]) -> tuple[Callable, int, AutodiffMode]:
        instance_id, arg_features = self.mapper.lookup(args)
        key = (self.func, instance_id, self.autodiff_mode)
        self.materialize(key=key, args=args, arg_features=arg_features)
        return key

    # For small kernels (< 3us), the performance can be pretty sensitive to overhead in __call__
    # Thus this part needs to be fast. (i.e. < 3us on a 4 GHz x64 CPU)
    @_shell_pop_print
    def __call__(self, *args, **kwargs) -> Any:
        args = _process_args(self, is_func=False, args=args, kwargs=kwargs)

        # Transform the primal kernel to forward mode grad kernel
        # then recover to primal when exiting the forward mode manager
        if self.runtime.fwd_mode_manager and not self.runtime.grad_replaced:
            # TODO: if we would like to compute 2nd-order derivatives by forward-on-reverse in a nested context manager fashion,
            # i.e., a `Tape` nested in the `FwdMode`, we can transform the kernels with `mode_original == AutodiffMode.REVERSE` only,
            # to avoid duplicate computation for 1st-order derivatives
            self.runtime.fwd_mode_manager.insert(self)

        # Both the class kernels and the plain-function kernels are unified now.
        # In both cases, |self.grad| is another Kernel instance that computes the
        # gradient. For class kernels, args[0] is always the kernel owner.

        # No need to capture grad kernels because they are already bound with their primal kernels
        if (
            self.autodiff_mode in (AutodiffMode.NONE, AutodiffMode.VALIDATION)
            and self.runtime.target_tape
            and not self.runtime.grad_replaced
        ):
            self.runtime.target_tape.insert(self, args)

        if self.autodiff_mode != AutodiffMode.NONE and impl.current_cfg().opt_level == 0:
            _logging.warn("""opt_level = 1 is enforced to enable gradient computation.""")
            impl.current_cfg().opt_level = 1
        key = self.ensure_compiled(*args)
        kernel_cpp = self.materialized_kernels[key]
        compiled_kernel_data = self.compiled_kernel_data_by_key.get(key, None)
        return self.launch_kernel(kernel_cpp, compiled_kernel_data, *args)


# For a GsTaichi class definition like below:
#
# @ti.data_oriented
# class X:
#   @ti.kernel
#   def foo(self):
#     ...
#
# When ti.kernel runs, the stackframe's |code_context| of Python 3.8(+) is
# different from that of Python 3.7 and below. In 3.8+, it is 'class X:',
# whereas in <=3.7, it is '@ti.data_oriented'. More interestingly, if the class
# inherits, i.e. class X(object):, then in both versions, |code_context| is
# 'class X(object):'...
_KERNEL_CLASS_STACKFRAME_STMT_RES = [
    re.compile(r"@(\w+\.)?data_oriented"),
    re.compile(r"class "),
]


def _inside_class(level_of_class_stackframe: int) -> bool:
    try:
        maybe_class_frame = sys._getframe(level_of_class_stackframe)
        statement_list = inspect.getframeinfo(maybe_class_frame)[3]
        if statement_list is None:
            return False
        first_statment = statement_list[0].strip()
        for pat in _KERNEL_CLASS_STACKFRAME_STMT_RES:
            if pat.match(first_statment):
                return True
    except:
        pass
    return False


def _kernel_impl(_func: Callable, level_of_class_stackframe: int, verbose: bool = False) -> GsTaichiCallable:
    # Can decorators determine if a function is being defined inside a class?
    # https://stackoverflow.com/a/8793684/12003165
    is_classkernel = _inside_class(level_of_class_stackframe + 1)

    if verbose:
        print(f"kernel={_func.__name__} is_classkernel={is_classkernel}")
    primal = Kernel(_func, autodiff_mode=AutodiffMode.NONE, _classkernel=is_classkernel)
    adjoint = Kernel(_func, autodiff_mode=AutodiffMode.REVERSE, _classkernel=is_classkernel)
    # Having |primal| contains |grad| makes the tape work.
    primal.grad = adjoint

    wrapped: GsTaichiCallable
    if is_classkernel:
        # For class kernels, their primal/adjoint callables are constructed
        # when the kernel is accessed via the instance inside
        # _BoundedDifferentiableMethod.
        # This is because we need to bind the kernel or |grad| to the instance
        # owning the kernel, which is not known until the kernel is accessed.
        #
        # See also: _BoundedDifferentiableMethod, data_oriented.
        @functools.wraps(_func)
        def wrapped_classkernel(*args, **kwargs):
            # If we reach here (we should never), it means the class is not decorated
            # with @ti.data_oriented, otherwise getattr would have intercepted the call.
            clsobj = type(args[0])
            assert not hasattr(clsobj, "_data_oriented")
            raise GsTaichiSyntaxError(f"Please decorate class {clsobj.__name__} with @ti.data_oriented")

        wrapped = GsTaichiCallable(_func, wrapped_classkernel)
    else:

        @functools.wraps(_func)
        def wrapped_func(*args, **kwargs):
            try:
                return primal(*args, **kwargs)
            except (GsTaichiCompilationError, GsTaichiRuntimeError) as e:
                if impl.get_runtime().print_full_traceback:
                    raise e
                raise type(e)("\n" + str(e)) from None

        wrapped = GsTaichiCallable(_func, wrapped_func)
        wrapped.grad = adjoint

    wrapped._is_wrapped_kernel = True
    wrapped._is_classkernel = is_classkernel
    wrapped._primal = primal
    wrapped._adjoint = adjoint
    primal.gstaichi_callable = wrapped
    return wrapped


def kernel(fn: Callable):
    """Marks a function as a GsTaichi kernel.

    A GsTaichi kernel is a function written in Python, and gets JIT compiled by
    GsTaichi into native CPU/GPU instructions (e.g. a series of CUDA kernels).
    The top-level ``for`` loops are automatically parallelized, and distributed
    to either a CPU thread pool or massively parallel GPUs.

    Kernel's gradient kernel would be generated automatically by the AutoDiff system.

    See also https://docs.taichi-lang.org/docs/syntax#kernel.

    Args:
        fn (Callable): the Python function to be decorated

    Returns:
        Callable: The decorated function

    Example::

        >>> x = ti.field(ti.i32, shape=(4, 8))
        >>>
        >>> @ti.kernel
        >>> def run():
        >>>     # Assigns all the elements of `x` in parallel.
        >>>     for i in x:
        >>>         x[i] = i
    """
    return _kernel_impl(fn, level_of_class_stackframe=3)


class _BoundedDifferentiableMethod:
    def __init__(self, kernel_owner: Any, wrapped_kernel_func: GsTaichiCallable | BoundGsTaichiCallable):
        clsobj = type(kernel_owner)
        if not getattr(clsobj, "_data_oriented", False):
            raise GsTaichiSyntaxError(f"Please decorate class {clsobj.__name__} with @ti.data_oriented")
        self._kernel_owner = kernel_owner
        self._primal = wrapped_kernel_func._primal
        self._adjoint = wrapped_kernel_func._adjoint
        self._is_staticmethod = wrapped_kernel_func._is_staticmethod
        self.__name__: str | None = None

    def __call__(self, *args, **kwargs):
        try:
            assert self._primal is not None
            if self._is_staticmethod:
                return self._primal(*args, **kwargs)
            return self._primal(self._kernel_owner, *args, **kwargs)

        except (GsTaichiCompilationError, GsTaichiRuntimeError) as e:
            if impl.get_runtime().print_full_traceback:
                raise e
            raise type(e)("\n" + str(e)) from None

    def grad(self, *args, **kwargs) -> Kernel:
        assert self._adjoint is not None
        return self._adjoint(self._kernel_owner, *args, **kwargs)


def data_oriented(cls):
    """Marks a class as GsTaichi compatible.

    To allow for modularized code, GsTaichi provides this decorator so that
    GsTaichi kernels can be defined inside a class.

    See also https://docs.taichi-lang.org/docs/odop

    Example::

        >>> @ti.data_oriented
        >>> class TiArray:
        >>>     def __init__(self, n):
        >>>         self.x = ti.field(ti.f32, shape=n)
        >>>
        >>>     @ti.kernel
        >>>     def inc(self):
        >>>         for i in self.x:
        >>>             self.x[i] += 1.0
        >>>
        >>> a = TiArray(32)
        >>> a.inc()

    Args:
        cls (Class): the class to be decorated

    Returns:
        The decorated class.
    """

    def _getattr(self, item):
        method = cls.__dict__.get(item, None)
        is_property = method.__class__ == property
        is_staticmethod = method.__class__ == staticmethod
        if is_property:
            x = method.fget
        else:
            x = super(cls, self).__getattribute__(item)
        if hasattr(x, "_is_wrapped_kernel"):
            if inspect.ismethod(x):
                wrapped = x.__func__
            else:
                wrapped = x
            assert isinstance(wrapped, (BoundGsTaichiCallable, GsTaichiCallable))
            wrapped._is_staticmethod = is_staticmethod
            if wrapped._is_classkernel:
                ret = _BoundedDifferentiableMethod(self, wrapped)
                ret.__name__ = wrapped.__name__  # type: ignore
                if is_property:
                    return ret()
                return ret
        if is_property:
            return x(self)
        return x

    cls.__getattribute__ = _getattr
    cls._data_oriented = True

    return cls


__all__ = ["data_oriented", "func", "kernel", "pyfunc", "real_func"]
