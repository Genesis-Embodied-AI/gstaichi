import ast
import inspect
import math
import re
import sys
import textwrap
import typing
import warnings
from dataclasses import (
    _FIELD,  # type: ignore[reportAttributeAccessIssue]
    _FIELDS,  # type: ignore[reportAttributeAccessIssue]
    is_dataclass,
)

# Must import 'partial' directly instead of the entire module to avoid attribute lookup overhead.
from functools import partial, update_wrapper, wraps
from typing import Any, Callable, DefaultDict, Type, TypeVar, cast, overload

# Must import 'ReferenceType' directly instead of the entire module to avoid attribute lookup overhead.
from weakref import ReferenceType

import numpy as np

from gstaichi._lib import core as _ti_core
from gstaichi._lib.core.gstaichi_python import (
    ASTBuilder,
    KernelLaunchContext,
)
from gstaichi.lang import _kernel_impl_dataclass, impl
from gstaichi.lang._ndarray import Ndarray
from gstaichi.lang._wrap_inspect import get_source_info_and_src
from gstaichi.lang.ast import (
    ASTTransformerContext,
)
from gstaichi.lang.exception import (
    GsTaichiCompilationError,
    GsTaichiRuntimeError,
    GsTaichiRuntimeTypeError,
    GsTaichiSyntaxError,
)
from gstaichi.lang.kernel_arguments import ArgMetadata
from gstaichi.lang.matrix import MatrixType
from gstaichi.lang.struct import StructType
from gstaichi.lang.util import cook_dtype, has_pytorch
from gstaichi.types import (
    ndarray_type,
    primitive_types,
    sparse_matrix_builder,
    template,
)
from gstaichi.types.enums import AutodiffMode, Layout
from gstaichi.types.utils import is_signed

from .._test_tools import warnings_helper
from .func import Func
from .gstaichi_callable import BoundGsTaichiCallable, GsTaichiCallable
from .kernel import Kernel
from .kernel_types import _KernelBatchedArgType

MAX_ARG_NUM = 512

# Define proxies for fast lookup
_NONE, _REVERSE = (
    AutodiffMode.NONE,
    AutodiffMode.REVERSE,
)
_ARG_EMPTY = inspect.Parameter.empty
_arch_cuda = _ti_core.Arch.cuda


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


def get_tree_and_ctx(
    self: "Func | Kernel",
    args: tuple[Any, ...],
    enforcing_dataclass_parameters: bool,
    # used_py_dataclass_parameters_enforcing: set[str] | None,
    excluded_parameters=(),
    is_kernel: bool = True,
    arg_features=None,
    ast_builder: ASTBuilder | None = None,
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
        # used_py_dataclass_parameters_collecting=current_kernel.used_py_dataclass_leaves_by_key_collecting[
        #     args_instance_key
        # ],
        enforcing_dataclass_parameters=enforcing_dataclass_parameters,
        # used_py_dataclass_parameters_enforcing=used_py_dataclass_parameters_enforcing,
    )
    return tree, ctx


def process_args(
    self: "Func | Kernel", is_pyfunc: bool, is_func: bool, args: tuple[Any, ...], kwargs
) -> tuple[Any, ...]:
    print("_process args is_func", is_func, "is_pyfunc", is_pyfunc, self.func)
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
            None,
            # current_kernel.used_py_dataclass_leaves_by_key_enforcing.get(currently_compiling_materialize_key),
            self.arg_metas,
        )
        # print("expanded arg metas expanded:", self.arg_metas_expanded)
    else:
        self.arg_metas_expanded = list(self.arg_metas)

    num_args = len(args)
    num_arg_metas = len(self.arg_metas_expanded)
    if num_args > num_arg_metas:
        arg_str = ", ".join(map(str, args))
        expected_str = ", ".join(f"{arg.name} : {arg.annotation}" for arg in self.arg_metas_expanded)
        msg_l = []
        msg_l.append(f"Too many arguments. Expected ({expected_str}), got ({arg_str}).")
        for i in range(num_args):
            if i < num_arg_metas:
                msg_l.append(f" - {i} arg meta: {self.arg_metas_expanded[i].name} arg type: {type(args[i])}")
            else:
                msg_l.append(f" - {i} arg meta: <out of arg metas> arg type: {type(args[i])}")
        msg_l.append(f"In function: {self.func}")
        raise GsTaichiSyntaxError("\n".join(msg_l))

    missing_arg_metas = self.arg_metas_expanded[num_args:]
    num_missing_args = len(missing_arg_metas)
    fused_args: list[Any] = [*args, *[arg_meta.default for arg_meta in missing_arg_metas]]
    # print("kernel_impl.py _process_args()")
    if kwargs:
        num_invalid_kwargs_args = len(kwargs)
        for i in range(num_args, num_arg_metas):
            arg_meta = self.arg_metas_expanded[i]
            value = kwargs.get(arg_meta.name, _ARG_EMPTY)
            # print("kwarg i", i, arg_meta, "value", value, type(value))
            if value is not _ARG_EMPTY:
                fused_args[i] = value
                num_invalid_kwargs_args -= 1
            elif fused_args[i] is _ARG_EMPTY:
                raise GsTaichiSyntaxError(f"Missing argument '{arg_meta.name}'.")
        if num_invalid_kwargs_args:
            for key, value in kwargs.items():
                for i, arg_meta in enumerate(self.arg_metas_expanded):
                    if key == arg_meta.name:
                        if i < num_args:
                            raise GsTaichiSyntaxError(f"Multiple values for argument '{key}'.")
                        break
                else:
                    raise GsTaichiSyntaxError(f"Unexpected argument '{key}'.")
    elif num_missing_args:
        for i in range(num_args, num_arg_metas):
            arg = fused_args[i]
            if fused_args[i] is _ARG_EMPTY:
                arg_meta = self.arg_metas_expanded[i]
                raise GsTaichiSyntaxError(f"Missing argument '{arg_meta.name}'.")

    return tuple(fused_args)


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


def cast_float(x: float | np.floating | np.integer | int) -> float:
    if not isinstance(x, (int, float, np.integer, np.floating)):
        raise ValueError(f"Invalid argument type '{type(x)}")
    return float(x)


def cast_int(x: int | np.integer) -> int:
    if not isinstance(x, (int, np.integer)):
        raise ValueError(f"Invalid argument type '{type(x)}")
    return int(x)


# Define proxies for fast lookup
_FLOAT, _INT, _UINT, _TI_ARRAY, _TI_ARRAY_WITH_GRAD = _KernelBatchedArgType


def destroy_callback(kernel_ref: ReferenceType["Kernel"], ref: ReferenceType):
    maybe_kernel = kernel_ref()
    if maybe_kernel is not None:
        maybe_kernel._launch_ctx_cache.clear()
        maybe_kernel._launch_ctx_cache_tracker.clear()
        maybe_kernel._prog_weakref = None


def recursive_set_args(
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
            num_args_, is_launch_ctx_cacheable_ = recursive_set_args(
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

    @wraps(_func)
    def wrapped_func(*args, **kwargs):
        try:
            return primal(*args, **kwargs)
        except (GsTaichiCompilationError, GsTaichiRuntimeError) as e:
            if impl.get_runtime().print_full_traceback:
                raise e
            raise type(e)("\n" + str(e)) from None

    wrapped: GsTaichiCallable
    if is_classkernel:
        # For class kernels, their primal/adjoint callables are constructed when the kernel is accessed via the
        # instance inside _BoundedDifferentiableMethod.
        # This is because we need to bind the kernel or |grad| to the instance owning the kernel, which is not known
        # until the kernel is accessed.
        # See also: _BoundedDifferentiableMethod, data_oriented.
        @wraps(_func)
        def wrapped_classkernel(*args, **kwargs):
            if args and not getattr(args[0], "_data_oriented", False):
                raise GsTaichiSyntaxError(f"Please decorate class {type(args[0]).__name__} with @ti.data_oriented")
            return wrapped_func(*args, **kwargs)

        wrapped = GsTaichiCallable(_func, wrapped_classkernel)
    else:
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
        self.__name__: str | None = None

    def __call__(self, *args, **kwargs):
        try:
            assert self._primal is not None
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

    def make_kernel_indirect(fun, is_property):
        @wraps(fun)
        def _kernel_indirect(self, *args, **kwargs):
            nonlocal fun
            ret = _BoundedDifferentiableMethod(self, fun)
            ret.__name__ = fun.__name__  # type: ignore
            return ret(*args, **kwargs)

        ret = GsTaichiCallable(fun, _kernel_indirect)
        if is_property:
            ret = property(ret)
        return ret

    # Iterate over all the attributes of the class to wrap member kernels in a way to ensure that they will be called
    # through _BoundedDifferentiableMethod. This extra layer of indirection is necessary to transparently forward the
    # owning instance to the primal function and its adjoint for auto-differentiation gradient computation.
    # There is a special treatment for properties, as they may actually hide kernels under the hood. In such a case,
    # the underlying function is extracted, wrapped as any member function, then wrapped again as a new property.
    # Note that all the other attributes can be left untouched.
    for name, attr in cls.__dict__.items():
        attr_type = type(attr)
        is_property = attr_type is property
        fun = attr.fget if is_property else attr
        if isinstance(fun, (BoundGsTaichiCallable, GsTaichiCallable)):
            if fun._is_wrapped_kernel:
                if fun._is_classkernel and attr_type is not staticmethod:
                    setattr(cls, name, make_kernel_indirect(fun, is_property))
    cls._data_oriented = True

    return cls


__all__ = ["data_oriented", "func", "kernel", "pyfunc", "real_func", "_KernelBatchedArgType"]
