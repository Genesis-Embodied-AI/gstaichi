import ast
import csv
import inspect
import json
import math
import os
import pathlib
import re
import sys
import textwrap
import time
import types
import typing
import warnings
from collections import defaultdict
from dataclasses import (
    _FIELD,  # type: ignore[reportAttributeAccessIssue]
    _FIELDS,  # type: ignore[reportAttributeAccessIssue]
    dataclass,
    is_dataclass,
)
from enum import IntEnum

# Must import 'partial' directly instead of the entire module to avoid attribute lookup overhead.
from functools import partial, update_wrapper, wraps
from typing import Any, Callable, DefaultDict, Type, TypeAlias, TypeVar, cast, overload

# Must import 'ReferenceType' directly instead of the entire module to avoid attribute lookup overhead.
from weakref import ReferenceType

import numpy as np

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
from gstaichi.lang._ndarray import Ndarray
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
from gstaichi.lang.impl import Program
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
)
from gstaichi.types.compound_types import CompoundType
from gstaichi.types.enums import AutodiffMode, Layout
from gstaichi.types.utils import is_signed

from .._test_tools import warnings_helper

MAX_ARG_NUM = 512


# Define proxies for fast lookup
_NONE, _VALIDATION, _FORWARD, _REVERSE = (
    AutodiffMode.NONE,
    AutodiffMode.VALIDATION,
    AutodiffMode.FORWARD,
    AutodiffMode.REVERSE,
)
_arch_cuda = _ti_core.Arch.cuda

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
        self.is_pure: bool = False
        update_wrapper(self, fn)

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
        if is_dataclass(parameter.annotation):
            _kernel_impl_dataclass.populate_global_vars_from_dataclass(
                parameter_name,
                parameter.annotation,
                py_args[i],
                global_vars=global_vars,
            )


def _get_tree_and_ctx(
    self: "Func | Kernel",
    args: tuple[Any, ...],
    used_py_dataclass_parameters_enforcing: set[str] | None,
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

    if current_kernel is not None:  # Kernel
        current_kernel.kernel_function_info = function_source_info
    if current_kernel is None:
        current_kernel = impl.get_runtime()._current_kernel
    assert current_kernel is not None
    current_kernel.visited_functions.add(function_source_info)

    autodiff_mode = current_kernel.autodiff_mode

    gstaichi_callable = current_kernel.gstaichi_callable
    is_pure = gstaichi_callable is not None and gstaichi_callable.is_pure
    global_vars = _get_global_vars(self.func)

    template_vars = {}
    if is_kernel or is_real_function:
        _populate_global_vars_for_templates(
            template_slot_locations=self.template_slot_locations,
            argument_metas=self.arg_metas,
            global_vars=template_vars,
            fn=self.func,
            py_args=args,
        )

    raise_on_templated_floats = impl.current_cfg().raise_on_templated_floats

    args_instance_key = current_kernel.currently_compiling_materialize_key
    assert args_instance_key is not None
    ctx = ASTTransformerContext(
        excluded_parameters=excluded_parameters,
        is_kernel=is_kernel,
        is_pure=is_pure,
        func=self,
        arg_features=arg_features,
        global_vars=global_vars,
        template_vars=template_vars,
        argument_data=args,
        src=src,
        start_lineno=function_source_info.start_lineno,
        end_lineno=function_source_info.end_lineno,
        file=function_source_info.filepath,
        ast_builder=ast_builder,
        is_real_function=is_real_function,
        autodiff_mode=autodiff_mode,
        raise_on_templated_floats=raise_on_templated_floats,
        used_py_dataclass_parameters_collecting=current_kernel.used_py_dataclass_leaves_by_key_collecting[
            args_instance_key
        ],
        used_py_dataclass_parameters_enforcing=used_py_dataclass_parameters_enforcing,
    )
    return tree, ctx


def _process_args(
    self: "Func | Kernel", is_pyfunc: bool, is_func: bool, args: tuple[Any, ...], kwargs
) -> tuple[Any, ...]:
    if is_func and not is_pyfunc:
        if typing.TYPE_CHECKING:
            assert isinstance(self, Func)
        current_kernel = self.current_kernel
        if typing.TYPE_CHECKING:
            assert current_kernel is not None
        currently_compiling_materialize_key = current_kernel.currently_compiling_materialize_key
        if typing.TYPE_CHECKING:
            assert currently_compiling_materialize_key is not None
        self.arg_metas_expanded = _kernel_impl_dataclass.expand_func_arguments(
            current_kernel.used_py_dataclass_leaves_by_key_enforcing.get(currently_compiling_materialize_key),
            self.arg_metas,
        )
    else:
        self.arg_metas_expanded = list(self.arg_metas)

    fused_args: list[Any] = [arg_meta.default for arg_meta in self.arg_metas_expanded]
    len_args = len(args)

    if len_args > len(fused_args):
        arg_str = ", ".join(map(str, args))
        expected_str = ", ".join(f"{arg.name} : {arg.annotation}" for arg in self.arg_metas_expanded)
        msg_l = []
        msg_l.append(f"Too many arguments. Expected ({expected_str}), got ({arg_str}).")
        for i in range(len_args):
            if i < len(self.arg_metas_expanded):
                msg_l.append(f" - {i} arg meta: {self.arg_metas_expanded[i].name} arg type: {type(args[i])}")
            else:
                msg_l.append(f" - {i} arg meta: <out of arg metas> arg type: {type(args[i])}")
        msg_l.append(f"In function: {self.func}")
        raise GsTaichiSyntaxError("\n".join(msg_l))

    for i, arg in enumerate(args):
        fused_args[i] = arg

    for key, value in kwargs.items():
        for i, arg in enumerate(self.arg_metas_expanded):
            if key == arg.name:
                if i < len_args:
                    raise GsTaichiSyntaxError(f"Multiple values for argument '{key}'.")
                fused_args[i] = value
                break
        else:
            raise GsTaichiSyntaxError(f"Unexpected argument '{key}'.")

    missing_parameters = []
    for i, arg in enumerate(fused_args):
        if arg is inspect.Parameter.empty:
            if self.arg_metas_expanded[i].annotation is inspect._empty:
                missing_parameters.append(f"Parameter `{self.arg_metas_expanded[i].name}` missing.")
            else:
                missing_parameters.append(
                    f"Parameter `{self.arg_metas_expanded[i].name} : {self.arg_metas_expanded[i].annotation}` missing."
                )
    if len(missing_parameters) > 0:
        msg_l = []
        msg_l.append("Error: missing parameters.")
        msg_l.extend(missing_parameters)
        msg_l.append("")
        msg_l.append("Debug info follows.")
        msg_l.append("fused args:")
        for i, arg in enumerate(fused_args):
            msg_l.append(f"  {i} {arg}")
        msg_l.append("arg metas:")
        for i, arg in enumerate(self.arg_metas_expanded):
            msg_l.append(f"  {i} {arg}")
        raise GsTaichiSyntaxError("\n".join(msg_l))

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
        self.arg_metas_expanded: list[ArgMetadata] = []
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
        self.current_kernel: Kernel | None = None

    def __call__(self: "Func", *args, **kwargs) -> Any:
        self.current_kernel = impl.get_runtime().current_kernel if impl.inside_kernel() else None
        args = _process_args(self, is_func=True, is_pyfunc=self.pyfunc, args=args, kwargs=kwargs)

        if not impl.inside_kernel():
            if not self.pyfunc:
                raise GsTaichiSyntaxError("GsTaichi functions cannot be called from Python-scope.")
            return self.func(*args)

        assert self.current_kernel is not None

        if self.is_real_function:
            if self.current_kernel.autodiff_mode != _NONE:
                self.current_kernel = None
                raise GsTaichiSyntaxError("Real function in gradient kernels unsupported.")
            instance_id, arg_features = self.mapper.lookup(impl.current_cfg().raise_on_templated_floats, args)
            key = _ti_core.FunctionKey(self.func.__name__, self.func_id, instance_id)
            if key.instance_id not in self.compiled:
                self.do_compile(key=key, args=args, arg_features=arg_features)
            self.current_kernel = None
            return self.func_call_rvalue(key=key, args=args)
        current_args_key = self.current_kernel.currently_compiling_materialize_key
        assert current_args_key is not None
        used_by_dataclass_parameters_enforcing = self.current_kernel.used_py_dataclass_leaves_by_key_enforcing.get(
            current_args_key
        )
        tree, ctx = _get_tree_and_ctx(
            self,
            is_kernel=False,
            args=args,
            ast_builder=self.current_kernel.ast_builder(),
            is_real_function=self.is_real_function,
            used_py_dataclass_parameters_enforcing=used_by_dataclass_parameters_enforcing,
        )

        struct_locals = _kernel_impl_dataclass.extract_struct_locals_from_context(ctx)

        tree = _kernel_impl_dataclass.unpack_ast_struct_expressions(tree, struct_locals=struct_locals)
        ret = transform_tree(tree, ctx)
        self.current_kernel = None
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
            self,
            is_kernel=False,
            args=args,
            arg_features=arg_features,
            is_real_function=self.is_real_function,
            used_py_dataclass_parameters_enforcing=None,
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
                annotation_type = type(annotation)
                if annotation_type is ndarray_type.NdarrayType:
                    pass
                elif issubclass(annotation_type, MatrixType):
                    pass
                elif annotation_type is StructType:
                    pass
                elif id(annotation) in primitive_types.type_ids:
                    pass
                elif annotation_type is template or annotation is template:
                    pass
                elif annotation_type is primitive_types.RefType:
                    pass
                elif annotation_type is type and is_dataclass(annotation):
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


@dataclass
class SrcLlCacheObservations:
    cache_key_generated: bool = False
    cache_validated: bool = False
    cache_loaded: bool = False
    cache_stored: bool = False


@dataclass
class FeLlCacheObservations:
    cache_hit: bool = False


def cast_float(x: float | np.floating | np.integer | int) -> float:
    if not isinstance(x, (int, float, np.integer, np.floating)):
        raise ValueError(f"Invalid argument type '{type(x)}")
    return float(x)


def cast_int(x: int | np.integer) -> int:
    if not isinstance(x, (int, np.integer)):
        raise ValueError(f"Invalid argument type '{type(x)}")
    return int(x)


class _KernelBatchedArgType(IntEnum):
    FLOAT = 0
    INT = 1
    UINT = 2
    TI_ARRAY = 3
    TI_ARRAY_WITH_GRAD = 4


# Define proxies for fast lookup
_FLOAT, _INT, _UINT, _TI_ARRAY, _TI_ARRAY_WITH_GRAD = _KernelBatchedArgType


ArgsHash: TypeAlias = tuple[int, ...]


@dataclass
class LaunchStats:
    kernel_args_count_by_type: dict[_KernelBatchedArgType, int]


def _destroy_callback(kernel_ref: ReferenceType["Kernel"], ref: ReferenceType):
    maybe_kernel = kernel_ref()
    if maybe_kernel is not None:
        maybe_kernel._launch_ctx_cache.clear()
        maybe_kernel._launch_ctx_cache_tracker.clear()
        maybe_kernel._prog_weakref = None


def _recursive_set_args(
    used_py_dataclass_parameters: set[tuple[str, ...]],
    py_dataclass_basename: tuple[str, ...],
    launch_ctx: KernelLaunchContext,
    launch_ctx_buffer: DefaultDict[_KernelBatchedArgType, list[tuple]],
    needed_arg_type: Type,
    provided_arg_type: Type,
    v: Any,
    index: int,
    actual_argument_slot: int,
    callbacks: list[Callable[[], Any]],
) -> tuple[int, bool]:
    """
    This function processes all the input python-side arguments of a given kernel so as to add them to the current
    launch context of a given kernel. Apart from a few exceptions, no call is made to the launch context directly,
    but rather accumulated in a buffer to be called all at once in a later stage. This avoid accumulating pybind11
    overhead for every single argument.

    Returns the number of underlying kernel args being set for a given Python arg, and whether the launch context
    buffer can be cached (see 'launch_kernel' for details).

    Note that templates don't set kernel args, and a single scalar, an external array (numpy or torch) or a taichi
    ndarray all set 1 kernel arg. Similarlty, a struct of N ndarrays would set N kernel args.
    """
    if actual_argument_slot >= MAX_ARG_NUM:
        raise GsTaichiRuntimeError(
            f"The number of elements in kernel arguments is too big! Do not exceed {MAX_ARG_NUM} on "
            f"{_ti_core.arch_name(impl.current_cfg().arch)} backend."
        )
    actual_argument_slot += 1

    needed_arg_type_id = id(needed_arg_type)
    needed_arg_basetype = type(needed_arg_type)

    # Note: do not use sth like "needed == f32". That would be slow.
    if needed_arg_type_id in primitive_types.real_type_ids:
        if not isinstance(v, (float, int, np.floating, np.integer)):
            raise GsTaichiRuntimeTypeError.get((index,), needed_arg_type.to_string(), provided_arg_type)
        launch_ctx_buffer[_FLOAT].append((index, float(v)))
        return 1, False
    if needed_arg_type_id in primitive_types.integer_type_ids:
        if not isinstance(v, (int, np.integer)):
            raise GsTaichiRuntimeTypeError.get((index,), needed_arg_type.to_string(), provided_arg_type)
        if is_signed(cook_dtype(needed_arg_type)):
            launch_ctx_buffer[_INT].append((index, int(v)))
        else:
            launch_ctx_buffer[_UINT].append((index, int(v)))
        return 1, False
    needed_arg_fields = getattr(needed_arg_type, _FIELDS, None)
    if needed_arg_fields is not None:
        if provided_arg_type is not needed_arg_type:
            raise GsTaichiRuntimeError("needed", needed_arg_type, "!= provided", provided_arg_type)
        # A dataclass must be frozen to be compatible with caching
        is_launch_ctx_cacheable = needed_arg_type.__hash__ is not None
        idx = 0
        for field in needed_arg_fields.values():
            if field._field_type is not _FIELD:
                continue
            field_name = field.name
            field_full_name = py_dataclass_basename + (field_name,)
            if field_full_name not in used_py_dataclass_parameters:
                continue
            # Storing attribute in a temporary to avoid repeated attribute lookup (~20ns penalty)
            field_type = field.type
            assert not isinstance(field_type, str)
            field_value = getattr(v, field_name)
            num_args_, is_launch_ctx_cacheable_ = _recursive_set_args(
                used_py_dataclass_parameters,
                field_full_name,
                launch_ctx,
                launch_ctx_buffer,
                field_type,
                field_type,
                field_value,
                index + idx,
                actual_argument_slot,
                callbacks,
            )
            idx += num_args_
            is_launch_ctx_cacheable &= is_launch_ctx_cacheable_
        return idx, is_launch_ctx_cacheable
    if needed_arg_basetype is ndarray_type.NdarrayType and isinstance(v, Ndarray):
        v_primal = v.arr
        v_grad = v.grad.arr if v.grad else None
        if v_grad is None:
            launch_ctx_buffer[_TI_ARRAY].append((index, v_primal))
        else:
            launch_ctx_buffer[_TI_ARRAY_WITH_GRAD].append((index, v_primal, v_grad))
        return 1, True
    if needed_arg_basetype is ndarray_type.NdarrayType:
        # v is things like torch Tensor and numpy array
        # Not adding type for this, since adds additional dependencies
        #
        # Element shapes are already specialized in GsTaichi codegen.
        # The shape information for element dims are no longer needed.
        # Therefore we strip the element shapes from the shape vector,
        # so that it only holds "real" array shapes.
        is_soa = needed_arg_type.layout == Layout.SOA
        array_shape = v.shape
        if math.prod(array_shape) > np.iinfo(np.int32).max:
            warnings.warn("Ndarray index might be out of int32 boundary but int64 indexing is not supported yet.")
        needed_arg_dtype = needed_arg_type.dtype
        if needed_arg_dtype is None or id(needed_arg_dtype) in primitive_types.type_ids:
            element_dim = 0
        else:
            element_dim = needed_arg_dtype.ndim
            array_shape = v.shape[element_dim:] if is_soa else v.shape[:-element_dim]
        if isinstance(v, np.ndarray):
            # Check ndarray flags is expensive (~250ns), so it is important to order branches according to hit stats
            if v.flags.c_contiguous:
                pass
            elif v.flags.f_contiguous:
                # TODO: A better way that avoids copying is saving strides info.
                v_contiguous = np.ascontiguousarray(v)
                v, v_orig_np = v_contiguous, v
                callbacks.append(partial(np.copyto, v_orig_np, v))
            else:
                raise ValueError(
                    "Non contiguous numpy arrays are not supported, please call np.ascontiguousarray(arr) "
                    "before passing it into gstaichi kernel."
                )
            launch_ctx.set_arg_external_array_with_shape(index, int(v.ctypes.data), v.nbytes, array_shape, 0)
        elif has_pytorch():
            import torch  # pylint: disable=C0415

            if isinstance(v, torch.Tensor):
                if not v.is_contiguous():
                    raise ValueError(
                        "Non contiguous tensors are not supported, please call tensor.contiguous() before "
                        "passing it into gstaichi kernel."
                    )
                gstaichi_arch = impl.current_cfg().arch

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
                            "Non contiguous gradient tensors are not supported, please call tensor.grad.contiguous() "
                            "before passing it into gstaichi kernel."
                        )

                grad = v.grad
                if (v.device.type != "cpu") and not (v.device.type == "cuda" and gstaichi_arch == _arch_cuda):
                    # For a torch tensor to be passed as as input argument (in and/or out) of a taichi kernel, its
                    # memory must be hosted either on CPU, or on CUDA if and only if GsTaichi is using CUDA backend.
                    # We just replace it with a CPU tensor and by the end of kernel execution we'll use the callback
                    # to copy the values back to the original tensor.
                    v_cpu = v.to(device="cpu")
                    v, v_orig_tc = v_cpu, v
                    callbacks.append(partial(v_orig_tc.data.copy_, v))
                    if grad is not None:
                        grad_cpu = grad.to(device="cpu")
                        grad, grad_orig = grad_cpu, grad
                        callbacks.append(partial(grad_orig.data.copy_, grad))

                launch_ctx.set_arg_external_array_with_shape(
                    index,
                    int(v.data_ptr()),
                    v.element_size() * v.nelement(),
                    array_shape,
                    int(grad.data_ptr()) if grad is not None else 0,
                )
            else:
                raise GsTaichiRuntimeTypeError(
                    f"Argument of type {type(v)} cannot be converted into required type {needed_arg_type}"
                )
        else:
            raise GsTaichiRuntimeTypeError(f"Argument {needed_arg_type} cannot be converted into required type {v}")
        return 1, False
    if issubclass(needed_arg_basetype, MatrixType):
        cast_func: Callable[[Any], int | float] | None = None
        if needed_arg_type.dtype in primitive_types.real_types:
            cast_func = cast_float
        elif needed_arg_type.dtype in primitive_types.integer_types:
            cast_func = cast_int
        else:
            raise ValueError(f"Matrix dtype {needed_arg_type.dtype} is not integer type or real type.")

        try:
            if needed_arg_type.ndim == 2:
                v = [cast_func(v[i, j]) for i in range(needed_arg_type.n) for j in range(needed_arg_type.m)]
            else:
                v = [cast_func(v[i]) for i in range(needed_arg_type.n)]
        except ValueError as e:
            raise GsTaichiRuntimeTypeError(
                f"Argument cannot be converted into required type {needed_arg_type.dtype}"
            ) from e

        v = needed_arg_type(*v)
        needed_arg_type.set_kernel_struct_args(v, launch_ctx, (index,))
        return 1, False
    if needed_arg_basetype is StructType:
        # Unclear how to make the following pass typing checks StructType implements __instancecheck__,
        # which should be a classmethod, but is currently an instance method.
        # TODO: look into this more deeply at some point
        if not isinstance(v, needed_arg_type):  # type: ignore
            raise GsTaichiRuntimeTypeError(
                f"Argument {provided_arg_type} cannot be converted into required type {needed_arg_type}"
            )
        needed_arg_type.set_kernel_struct_args(v, launch_ctx, (index,))
        return 1, False
    if needed_arg_type is template or needed_arg_basetype is template:
        return 0, True
    if needed_arg_basetype is sparse_matrix_builder:
        # Pass only the base pointer of the ti.types.sparse_matrix_builder() argument
        launch_ctx_buffer[_UINT].append((index, v._get_ndarray_addr()))
        return 1, True
    raise ValueError(f"Argument type mismatch. Expecting {needed_arg_type}, got {type(v)}.")


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
        self.arg_metas_expanded: list[ArgMetadata] = []
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
        self._last_compiled_kernel_data: CompiledKernelData | None = None  # for dev/debug
        # for collecting, we'll grab an empty set if it doesnt exist
        self.used_py_dataclass_leaves_by_key_collecting: dict[CompiledKernelKeyType, set[str]] = defaultdict(set)
        # however, for enforcing, we want None if it doesn't exist (we'll use .get() instead of [] )
        self.used_py_dataclass_leaves_by_key_enforcing: dict[CompiledKernelKeyType, set[str]] = {}
        self.used_py_dataclass_leaves_by_key_enforcing_dotted: dict[CompiledKernelKeyType, set[tuple[str, ...]]] = {}
        self.currently_compiling_materialize_key: CompiledKernelKeyType | None = None

        self.src_ll_cache_observations: SrcLlCacheObservations = SrcLlCacheObservations()
        self.fe_ll_cache_observations: FeLlCacheObservations = FeLlCacheObservations()

        # The cache key corresponds to the hash of the (packed) python-side input arguments of the kernel.
        # * '_launch_ctx_cache' is storing a backup of the launch context BEFORE ever calling the kernel.
        # * '_launch_ctx_cache_tracker' is used for bounding the lifetime of a cache entry to its corresponding set of
        #   input arguments. Internally, this is done by wrapping all Taichi ndarrays as weak reference.
        # * '_prog_weakref'is used for bounding the lifetime of the entire cache to the Taichi programm managing all
        #   the launch context being stored in cache.
        # See 'launch_kernel' for details regarding the intended use of caching.
        self._launch_ctx_cache: dict[ArgsHash, KernelLaunchContext] = {}
        self._launch_ctx_cache_tracker: dict[ArgsHash, list[ReferenceType]] = {}
        self._prog_weakref: ReferenceType[Program] | None = None

    def ast_builder(self) -> ASTBuilder:
        assert self.kernel_cpp is not None
        return self.kernel_cpp.ast_builder()

    def reset(self) -> None:
        self.runtime = impl.get_runtime()
        self.materialized_kernels = {}
        self.compiled_kernel_data_by_key = {}
        self._last_compiled_kernel_data = None
        self.src_ll_cache_observations = SrcLlCacheObservations()
        self.fe_ll_cache_observations = FeLlCacheObservations()
        self.used_py_dataclass_leaves_by_key_collecting = defaultdict(set)
        self.used_py_dataclass_leaves_by_key_enforcing = {}
        self.used_py_dataclass_leaves_by_key_enforcing_dotted = {}
        self.currently_compiling_materialize_key = None

    def extract_arguments(self) -> None:
        sig = inspect.signature(self.func)
        if sig.return_annotation not in {inspect._empty, None}:
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
                if isinstance(annotation, (template, ndarray_type.NdarrayType)):
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
                elif annotation is template:
                    pass
                elif isinstance(annotation, type) and is_dataclass(annotation):
                    pass
                else:
                    raise GsTaichiSyntaxError(f"Invalid type annotation (argument {i}) of Taichi kernel: {annotation}")
            self.arg_metas.append(ArgMetadata(annotation, param.name, param.default))

    def materialize(self, key: CompiledKernelKeyType | None, args: tuple[Any, ...], arg_features=None):
        if key is None:
            key = (self.func, 0, self.autodiff_mode)
        self.runtime.materialize()
        self.fast_checksum = None

        self.currently_compiling_materialize_key = key

        if key in self.materialized_kernels:
            return

        used_py_dataclass_parameters: set[str] | None = None

        if self.runtime.src_ll_cache and self.gstaichi_callable and self.gstaichi_callable.is_pure:
            kernel_source_info, _src = get_source_info_and_src(self.func)
            self.fast_checksum = src_hasher.create_cache_key(
                self.raise_on_templated_floats, kernel_source_info, args, self.arg_metas
            )
            if self.fast_checksum:
                self.src_ll_cache_observations.cache_key_generated = True
                used_py_dataclass_parameters = src_hasher.load(self.fast_checksum)
            if used_py_dataclass_parameters is not None:
                self.src_ll_cache_observations.cache_validated = True
                prog = impl.get_runtime().prog
                assert self.fast_checksum is not None
                self.compiled_kernel_data_by_key[key] = prog.load_fast_cache(
                    self.fast_checksum,
                    self.func.__name__,
                    prog.config(),
                    prog.get_device_caps(),
                )
                if self.compiled_kernel_data_by_key[key]:
                    self.src_ll_cache_observations.cache_loaded = True
                    self.used_py_dataclass_leaves_by_key_enforcing[key] = used_py_dataclass_parameters
                    self.used_py_dataclass_leaves_by_key_enforcing_dotted[key] = set(
                        [tuple(p.split("__ti_")[1:]) for p in used_py_dataclass_parameters]
                    )
        elif self.gstaichi_callable and not self.gstaichi_callable.is_pure and self.runtime.print_non_pure:
            # The bit in caps should not be modified without updating corresponding test
            # freetext can be freely modified.
            # As for why we are using `print` rather than eg logger.info, it is because
            # this is only printed when ti.init(print_non_pure=..) is True. And it is
            # confusing to set that to True, and see nothing printed.
            print(f"[NOT_PURE] Debug information: not pure: {self.func.__name__}")

        kernel_name = f"{self.func.__name__}_c{self.kernel_counter}_{key[1]}"
        _logging.trace(f"Materializing kernel {kernel_name} in {self.autodiff_mode}...")

        range_begin = 0 if used_py_dataclass_parameters is None else 1
        for _pass in range(range_begin, 2):
            used_py_dataclass_leaves_by_key_enforcing = None
            if _pass == 1:
                assert used_py_dataclass_parameters is not None
                used_py_dataclass_leaves_by_key_enforcing = set()
                for param in used_py_dataclass_parameters:
                    split_param = param.split("__ti_")
                    for i in range(len(split_param), 0, -1):
                        joined = "__ti_".join(split_param[:i])
                        if joined in used_py_dataclass_leaves_by_key_enforcing:
                            break
                        used_py_dataclass_leaves_by_key_enforcing.add(joined)
                self.used_py_dataclass_leaves_by_key_enforcing[key] = used_py_dataclass_leaves_by_key_enforcing
                self.used_py_dataclass_leaves_by_key_enforcing_dotted[key] = set(
                    [tuple(p.split("__ti_")[1:]) for p in used_py_dataclass_leaves_by_key_enforcing]
                )
            tree, ctx = _get_tree_and_ctx(
                self,
                args=args,
                excluded_parameters=self.template_slot_locations,
                arg_features=arg_features,
                current_kernel=self,
                used_py_dataclass_parameters_enforcing=used_py_dataclass_leaves_by_key_enforcing,
            )

            if self.autodiff_mode != _NONE:
                KernelSimplicityASTChecker(self.func).visit(tree)

            # Do not change the name of 'gstaichi_ast_generator'
            # The warning system needs this identifier to remove unnecessary messages
            def gstaichi_ast_generator(kernel_cxx: KernelCxx):
                nonlocal tree, used_py_dataclass_parameters
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

                    if os.environ.get("TI_DUMP_AST", "") == "1" and _pass == 1:
                        target_dir = pathlib.Path("/tmp/ast")
                        target_dir.mkdir(parents=True, exist_ok=True)

                        start = time.time()
                        ast_str = ast.dump(tree, indent=2)
                        output_file = target_dir / f"{kernel_name}_ast_.txt"
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
                    if not used_py_dataclass_parameters:
                        struct_locals = _kernel_impl_dataclass.extract_struct_locals_from_context(ctx)
                    else:
                        struct_locals = used_py_dataclass_parameters
                    tree = _kernel_impl_dataclass.unpack_ast_struct_expressions(tree, struct_locals=struct_locals)
                    ctx.only_parse_function_def = self.compiled_kernel_data_by_key.get(key) is not None
                    transform_tree(tree, ctx)
                    if not ctx.is_real_function and not ctx.only_parse_function_def:
                        if self.return_type and ctx.returned != ReturnStatus.ReturnedValue:
                            raise GsTaichiSyntaxError("Kernel has a return type but does not have a return statement")
                    used_py_dataclass_parameters = self.used_py_dataclass_leaves_by_key_collecting[key]
                finally:
                    self.runtime.inside_kernel = False
                    self.runtime._current_kernel = None
                    self.runtime._compiling_callable = None

            gstaichi_kernel = impl.get_runtime().prog.create_kernel(
                gstaichi_ast_generator, kernel_name, self.autodiff_mode
            )
            if _pass == 1:
                assert key not in self.materialized_kernels
                self.materialized_kernels[key] = gstaichi_kernel

    def launch_kernel(self, t_kernel: KernelCxx, compiled_kernel_data: CompiledKernelData | None, *args) -> Any:
        assert len(args) == len(self.arg_metas), f"{len(self.arg_metas)} arguments needed but {len(args)} provided"

        # Keep track of taichi runtime to automatically clear cache if destroyed
        if self._prog_weakref is None:
            prog = impl.get_runtime().prog
            assert prog is not None
            self._prog_weakref = ReferenceType(prog, partial(_destroy_callback, ReferenceType(self)))
        else:
            # Since we already store a weak reference to taichi program, it is much faster to use it rather than
            # paying the overhead of calling pybind11 functions (~200ns vs 5ns).
            prog = self._prog_weakref()
        assert prog is not None

        # Here, we are tracking whether a launch context buffer can be cached.
        # The point of caching the launch context buffer is allowing skipping recursive processing of all the input
        # arguments one-by-one, which is adding a significant overhead, without changing anything in regards of the
        # function calls to the launch context that must be made for a given kernel.
        # You can understand this as resolving the static part of the entire control flow of '_recursive_set_args'
        # for a given set of arguments, which is (mostly surely uniquely) characterized by its hash, gathering all
        # the instructions that cannot be evaluated statically and packing them in a buffer without evaluating them at
        # this point. This buffer is then cached once and for all and evaluated every time the exact same set of input
        # argument is passed. This means that, ultimately, it will result in the exact same function calls with or
        # without caching. In this particular case, the function calls corresponds to adding arguments to the current
        # context for this kernel call.
        # A launch context buffer is considered cache-friendly if and only if no direct call to the launch context
        # where made preemptively during the recursive processing of the arguments, all of leaves of the arguments are
        # pointers, the address of these pointers cannot change, and the set of leaves is fixed.
        # The lifetime of a cache entry is bound to the lifetime of any of its input arguments: the first being garbage
        # collected will invalidate the entire entry. Moreover, the entire cache registry is bound to the lifetime of
        # the taichi prog itself, which means that calling `ti.reset()` will automatically clear the cache. Note that
        # the cache stores wear references to pointers, so it does not hold alife any allocated memory.
        callbacks: list[Callable[[], None]] = []
        launch_ctx = t_kernel.make_launch_context()
        launch_ctx_cache: KernelLaunchContext | None = None
        launch_ctx_cache_tracker: list[ReferenceType] | None = None
        args_hash: ArgsHash = tuple(map(id, args))
        try:
            launch_ctx_cache_tracker = self._launch_ctx_cache_tracker[args_hash]
        except KeyError:
            pass
        if not launch_ctx_cache_tracker:  # Empty or none
            launch_ctx_buffer: DefaultDict[_KernelBatchedArgType, list[tuple]] = defaultdict(list)
            actual_argument_slot = 0
            is_launch_ctx_cacheable = True
            template_num = 0
            i_out = 0
            assert self.currently_compiling_materialize_key
            used_py_dataclass_parameters_enforcing_dotted = self.used_py_dataclass_leaves_by_key_enforcing_dotted[
                self.currently_compiling_materialize_key
            ]
            for i_in, val in enumerate(args):
                needed_ = self.arg_metas[i_in].annotation
                if needed_ is template or type(needed_) is template:
                    template_num += 1
                    i_out += 1
                    continue
                num_args_, is_launch_ctx_cacheable_ = _recursive_set_args(
                    used_py_dataclass_parameters_enforcing_dotted,
                    (self.arg_metas[i_in].name,),
                    launch_ctx,
                    launch_ctx_buffer,
                    needed_,
                    type(val),
                    val,
                    i_out - template_num,
                    actual_argument_slot,
                    callbacks,
                )
                i_out += num_args_
                is_launch_ctx_cacheable &= is_launch_ctx_cacheable_

            kernel_args_count_by_type = defaultdict(int)
            kernel_args_count_by_type.update(
                {key: len(launch_ctx_args) for key, launch_ctx_args in launch_ctx_buffer.items()}
            )
            self.launch_stats = LaunchStats(kernel_args_count_by_type=kernel_args_count_by_type)

            # All arguments to context in batches to mitigate overhead of calling Python bindings repeatedly.
            # This is essential because calling any pybind11 function is adding ~180ns penalty no matter what.
            # Note that we are allowed to do this because GsTaichi Launch Kernel context is storing the input
            # arguments in an unordered list. The actual runtime (gfx, llvm...) will later query this context
            # in correct order.
            if launch_ctx_args := launch_ctx_buffer.get(_FLOAT):
                launch_ctx.set_args_float(*zip(*launch_ctx_args))  # type: ignore
            if launch_ctx_args := launch_ctx_buffer.get(_INT):
                launch_ctx.set_args_int(*zip(*launch_ctx_args))  # type: ignore
            if launch_ctx_args := launch_ctx_buffer.get(_UINT):
                launch_ctx.set_args_uint(*zip(*launch_ctx_args))  # type: ignore
            if launch_ctx_args := launch_ctx_buffer.get(_TI_ARRAY):
                launch_ctx.set_args_ndarray(*zip(*launch_ctx_args))  # type: ignore
            if launch_ctx_args := launch_ctx_buffer.get(_TI_ARRAY_WITH_GRAD):
                launch_ctx.set_args_ndarray_with_grad(*zip(*launch_ctx_args))  # type: ignore

            if is_launch_ctx_cacheable and args_hash is not None:
                # TODO: It some rare occurrences, arguments can be cached yet not hashable. Ignoring for now...
                launch_ctx_cache = t_kernel.make_launch_context()
                launch_ctx_cache.copy(launch_ctx)
                self._launch_ctx_cache[args_hash] = launch_ctx_cache

                # Note that the clearing callback will only be called once despite being registered for each tracked
                # objects, because all the weakrefs get deallocated right away, and their respective callback
                # vanishes with them, without even getting a chance to get called. This means that registring the
                # clearing callback systematically does not incur any cumulative runtime penalty yet ensures full
                # memory safety.
                launch_ctx_cache_tracker_: list[ReferenceType] = []
                clear_callback = lambda ref: launch_ctx_cache_tracker_.clear()
                if launch_ctx_args := launch_ctx_buffer.get(_TI_ARRAY):
                    _, arrs = zip(*launch_ctx_args)
                    launch_ctx_cache_tracker_ += [ReferenceType(arr, clear_callback) for arr in arrs]
                if launch_ctx_args := launch_ctx_buffer.get(_TI_ARRAY_WITH_GRAD):
                    _, arrs, arrs_grad = zip(*launch_ctx_args)
                    launch_ctx_cache_tracker_ += [ReferenceType(arr, clear_callback) for arr in arrs]
                    launch_ctx_cache_tracker_ += [ReferenceType(arr_grad, clear_callback) for arr_grad in arrs_grad]
                self._launch_ctx_cache_tracker[args_hash] = launch_ctx_cache_tracker_
        else:
            assert args_hash is not None
            launch_ctx.copy(self._launch_ctx_cache[args_hash])

        try:
            if not compiled_kernel_data:
                # Store Taichi program config and device cap for efficiency because they are used at multiple places
                prog_config = prog.config()
                prog_device_cap = prog.get_device_caps()

                compile_result: CompileResult = prog.compile_kernel(prog_config, prog_device_cap, t_kernel)
                if os.environ.get("TI_DUMP_KERNEL_CHECKSUMS", "0") == "1":
                    debug_dump_path = pathlib.Path(impl.current_cfg().debug_dump_path)
                    checksums_file_path = debug_dump_path / "checksums.csv"
                    kernels_dump_dir = debug_dump_path / "kernels"
                    file_exists = checksums_file_path.exists()
                    if self.fast_checksum:
                        with checksums_file_path.open("a") as f:
                            dict_writer = csv.DictWriter(f, fieldnames=["kernel", "fe", "src"])
                            if not file_exists:
                                dict_writer.writeheader()
                            dict_writer.writerow(
                                {
                                    "kernel": self.func.__name__,
                                    "fe": compile_result.cache_key,
                                    "src": self.fast_checksum,
                                }
                            )
                            f.flush()
                        kernels_dump_dir.mkdir(exist_ok=True)
                        ch_ir_path = kernels_dump_dir / f"{compile_result.cache_key}.ll"
                        if not ch_ir_path.exists() and self.kernel_cpp:
                            with ch_ir_path.open("w") as f:
                                f.write(self.kernel_cpp.to_string())
                compiled_kernel_data = compile_result.compiled_kernel_data
                if compile_result.cache_hit:
                    self.fe_ll_cache_observations.cache_hit = True
                if self.fast_checksum:
                    assert self.currently_compiling_materialize_key is not None
                    src_hasher.store(
                        self.fast_checksum,
                        self.visited_functions,
                        self.used_py_dataclass_leaves_by_key_enforcing[self.currently_compiling_materialize_key],
                    )
                    prog.store_fast_cache(
                        self.fast_checksum,
                        self.kernel_cpp,
                        prog_config,
                        prog_device_cap,
                        compiled_kernel_data,
                    )
                    self.src_ll_cache_observations.cache_stored = True
            self._last_compiled_kernel_data = compiled_kernel_data
            prog.launch_kernel(compiled_kernel_data, launch_ctx)
        except Exception as e:
            e = handle_exception_from_cpp(e)
            if impl.get_runtime().print_full_traceback:
                raise e
            raise e from None

        for c in callbacks:
            c()

        self.currently_compiling_materialize_key = None

        return_type = self.return_type
        if return_type or self.has_print:
            runtime_ops.sync()

        if not return_type:
            return None
        if len(return_type) == 1:
            return self.construct_kernel_ret(launch_ctx, return_type[0], (0,))
        return tuple([self.construct_kernel_ret(launch_ctx, ret_type, (i,)) for i, ret_type in enumerate(return_type)])

    def construct_kernel_ret(self, launch_ctx: KernelLaunchContext, ret_type: Any, indices: tuple[int, ...]):
        if isinstance(ret_type, CompoundType):
            return ret_type.from_kernel_struct_ret(launch_ctx, indices)
        if ret_type in primitive_types.integer_types:
            if is_signed(cook_dtype(ret_type)):
                return launch_ctx.get_struct_ret_int(indices)
            return launch_ctx.get_struct_ret_uint(indices)
        if ret_type in primitive_types.real_types:
            return launch_ctx.get_struct_ret_float(indices)
        raise GsTaichiRuntimeTypeError(f"Invalid return type on index={indices}")

    def ensure_compiled(self, *args: tuple[Any, ...]) -> tuple[Callable, int, AutodiffMode]:
        try:
            instance_id, arg_features = self.mapper.lookup(self.raise_on_templated_floats, args)
        except Exception as e:
            raise type(e)(f"exception while trying to ensure compiled {self.func}:\n{e}") from e
        key = (self.func, instance_id, self.autodiff_mode)
        self.materialize(key=key, args=args, arg_features=arg_features)
        return key

    # For small kernels (< 3us), the performance can be pretty sensitive to overhead in __call__
    # Thus this part needs to be fast. (i.e. < 3us on a 4 GHz x64 CPU)
    @_shell_pop_print
    def __call__(self, *args, **kwargs) -> Any:
        self.raise_on_templated_floats = impl.current_cfg().raise_on_templated_floats

        args = _process_args(self, is_func=False, is_pyfunc=False, args=args, kwargs=kwargs)

        # Transform the primal kernel to forward mode grad kernel
        # then recover to primal when exiting the forward mode manager
        if self.runtime.fwd_mode_manager and not self.runtime.grad_replaced:
            # TODO: if we would like to compute 2nd-order derivatives by forward-on-reverse in a nested context manager
            # fashion, i.e., a `Tape` nested in the `FwdMode`, we can transform the kernels with
            # `mode_original == AutodiffMode.REVERSE` only, to avoid duplicate computation for 1st-order derivatives.
            self.runtime.fwd_mode_manager.insert(self)

        # Both the class kernels and the plain-function kernels are unified now.
        # In both cases, |self.grad| is another Kernel instance that computes the
        # gradient. For class kernels, args[0] is always the kernel owner.

        # No need to capture grad kernels because they are already bound with their primal kernels
        if self.autodiff_mode in (_NONE, _VALIDATION) and self.runtime.target_tape and not self.runtime.grad_replaced:
            self.runtime.target_tape.insert(self, args)

        if self.autodiff_mode != _NONE and impl.current_cfg().opt_level == 0:
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
    primal = Kernel(_func, autodiff_mode=_NONE, _classkernel=is_classkernel)
    adjoint = Kernel(_func, autodiff_mode=_REVERSE, _classkernel=is_classkernel)
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
        @wraps(_func)
        def wrapped_classkernel(*args, **kwargs):
            # If we reach here (we should never), it means the class is not decorated
            # with @ti.data_oriented, otherwise getattr would have intercepted the call.
            clsobj = type(args[0])
            assert not hasattr(clsobj, "_data_oriented")
            raise GsTaichiSyntaxError(f"Please decorate class {clsobj.__name__} with @ti.data_oriented")

        wrapped = GsTaichiCallable(_func, wrapped_classkernel)
    else:

        @wraps(_func)
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


F = TypeVar("F", bound=Callable[..., typing.Any])


@overload
# TODO: This callable should be Callable[[F], F].
# See comments below.
def kernel(_fn: None = None, *, pure: bool = False) -> Callable[[Any], Any]: ...


# TODO: This next overload should return F, but currently that will cause issues
# with ndarray type. We need to migrate ndarray type to be basically
# the actual Ndarray, with Generic types, rather than some other
# NdarrayType class. The _fn should also be F by the way.
# However, by making it return Any, we can make the pure parameter
# change now, without breaking pyright.
@overload
def kernel(_fn: Any, *, pure: bool = False) -> Any: ...


def kernel(_fn: Callable[..., typing.Any] | None = None, *, pure: bool | None = None, fastcache: bool = False):
    """
    Marks a function as a GsTaichi kernel.

    A GsTaichi kernel is a function written in Python, and gets JIT compiled by
    GsTaichi into native CPU/GPU instructions (e.g. a series of CUDA kernels).
    The top-level ``for`` loops are automatically parallelized, and distributed
    to either a CPU thread pool or massively parallel GPUs.

    Kernel's gradient kernel would be generated automatically by the AutoDiff system.

    Example::

        >>> x = ti.field(ti.i32, shape=(4, 8))
        >>>
        >>> @ti.kernel
        >>> def run():
        >>>     # Assigns all the elements of `x` in parallel.
        >>>     for i in x:
        >>>         x[i] = i
    """

    def decorator(fn: F, has_kernel_params: bool = True) -> F:
        # Adjust stack frame: +1 if called via decorator factory (@kernel()), else as-is (@kernel)
        if has_kernel_params:
            level = 3
        else:
            level = 4

        wrapped = _kernel_impl(fn, level_of_class_stackframe=level)
        wrapped.is_pure = pure is not None and pure or fastcache
        if pure is not None:
            warnings_helper.warn_once(
                "@ti.kernel parameter `pure` is deprecated. Please use parameter `fastcache`. "
                "`pure` parameter is intended to be removed in 4.0.0"
            )

        update_wrapper(wrapped, fn)
        return cast(F, wrapped)

    if _fn is None:
        # Called with @kernel() or @kernel(foo="bar")
        return decorator

    return decorator(_fn, has_kernel_params=False)


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
    # Backup the original attribute getter before overwriting it.
    # Note that this is faster, and more rigorous, to use this pattern over calling the parent method, ie
    # `super(cls, self).__getattribute__`. Faster because it avoid relying on MRO at runtime, more rigorous
    # because calling the parent method will by-pass the method `__getattribute__` that may already be
    # overloaded by the original class.
    getattribute_orig = cls.__getattribute__

    def _getattr(self, item):
        nonlocal getattribute_orig

        method = cls.__dict__.get(item)
        method_type = type(method)
        is_property = method_type is property
        if is_property:
            x = method.fget
        else:
            x = getattribute_orig(self, item)
        if hasattr(x, "_is_wrapped_kernel"):
            if inspect.ismethod(x):
                wrapped = x.__func__
            else:
                wrapped = x
            assert isinstance(wrapped, (BoundGsTaichiCallable, GsTaichiCallable))
            wrapped._is_staticmethod = method_type is staticmethod
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


__all__ = ["data_oriented", "func", "kernel", "pyfunc", "real_func", "_KernelBatchedArgType"]
