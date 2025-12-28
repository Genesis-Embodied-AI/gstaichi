from typing import TYPE_CHECKING, Any, Callable

from gstaichi._lib import core as _ti_core
from gstaichi._lib.core.gstaichi_python import (
    Function as FunctionCxx,
)
from gstaichi._lib.core.gstaichi_python import FunctionKey
from gstaichi.lang import _kernel_impl_dataclass, impl, ops
from gstaichi.lang.any_array import AnyArray
from gstaichi.lang.ast import (
    transform_tree,
)
from gstaichi.lang.ast.ast_transformer_utils import ReturnStatus
from gstaichi.lang.exception import (
    GsTaichiSyntaxError,
    GsTaichiTypeError,
)
from gstaichi.lang.expr import Expr
from gstaichi.lang.matrix import MatrixType
from gstaichi.lang.struct import StructType
from gstaichi.types import (
    ndarray_type,
    primitive_types,
    template,
)
from gstaichi.types.enums import AutodiffMode

from ._func_base import FuncBase

if TYPE_CHECKING:
    from .kernel import Kernel

# Define proxy for fast lookup
_NONE = AutodiffMode.NONE


class Func(FuncBase):
    function_counter = 0

    def __init__(self, _func: Callable, _classfunc=False, _pyfunc=False, is_real_function=False) -> None:
        super().__init__(
            func=_func,
            is_classfunc=_classfunc,
            is_kernel=False,
            is_classkernel=False,
            is_real_function=is_real_function,
        )
        self.func_id = Func.function_counter
        Func.function_counter += 1
        self.compiled: dict[int, Callable] = {}  # only for real funcs
        self.classfunc = _classfunc
        self.pyfunc = _pyfunc
        self.is_real_function = is_real_function
        self.cxx_function_by_id: dict[int, FunctionCxx] = {}
        self.has_print = False
        # Used during compilation. Assumes only one compilation at a time (single-threaded).
        self.current_kernel: Kernel | None = None

    def __call__(self: "Func", *py_args, **kwargs) -> Any:
        self.current_kernel = impl.get_runtime().current_kernel if impl.inside_kernel() else None
        py_args = self.fuse_args(is_func=True, is_pyfunc=self.pyfunc, py_args=py_args, kwargs=kwargs)

        if not impl.inside_kernel():
            if not self.pyfunc:
                raise GsTaichiSyntaxError("GsTaichi functions cannot be called from Python-scope.")
            return self.func(*py_args)

        assert self.current_kernel is not None

        if self.is_real_function:
            if self.current_kernel.autodiff_mode != _NONE:
                self.current_kernel = None
                raise GsTaichiSyntaxError("Real function in gradient kernels unsupported.")
            instance_id, arg_features = self.mapper.lookup(impl.current_cfg().raise_on_templated_floats, py_args)
            key = FunctionKey(self.func.__name__, self.func_id, instance_id)
            if key.instance_id not in self.compiled:
                self.do_compile(key=key, args=py_args, arg_features=arg_features)
            self.current_kernel = None
            return self.func_call_rvalue(key=key, args=py_args)
        current_args_key = self.current_kernel.currently_compiling_materialize_key
        assert current_args_key is not None
        used_by_dataclass_parameters_enforcing = self.current_kernel.used_py_dataclass_leaves_by_key_enforcing.get(
            current_args_key
        )
        tree, ctx = self.get_tree_and_ctx(
            is_kernel=False,
            py_args=py_args,
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
            self.cxx_function_by_id[key.instance_id], non_template_args, dbg_info
        )
        if self.return_type is None:
            return None
        func_call = Expr(func_call)
        ret = []

        for i, return_type in enumerate(self.return_type):
            if id(return_type) in primitive_types.type_ids:
                ret.append(Expr(_ti_core.make_get_element_expr(func_call.ptr, (i,), dbg_info)))
            elif isinstance(return_type, (StructType, MatrixType)):
                ret.append(return_type.from_gstaichi_object(func_call, (i,)))
            else:
                raise GsTaichiTypeError(f"Unsupported return type for return value {i}: {return_type}")
        if len(ret) == 1:
            return ret[0]
        return tuple(ret)

    def do_compile(self, key: FunctionKey, args: tuple[Any, ...], arg_features: tuple[Any, ...]) -> None:
        """
        only for real func
        """
        tree, ctx = self.get_tree_and_ctx(
            is_kernel=False,
            py_args=args,
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

        self.cxx_function_by_id[key.instance_id] = fn
        self.compiled[key.instance_id] = func_body
        self.cxx_function_by_id[key.instance_id].set_function_body(func_body)
