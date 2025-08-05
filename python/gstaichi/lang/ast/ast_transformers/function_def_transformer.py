# type: ignore

import ast
import dataclasses
from typing import Any, Callable

from gstaichi.lang import (
    _ndarray,
    any_array,
    expr,
    impl,
    kernel_arguments,
    matrix,
)
from gstaichi.lang import ops as ti_ops
from gstaichi.lang.argpack import ArgPackType
from gstaichi.lang.ast.ast_transformer_utils import (
    ASTTransformerContext,
)
from gstaichi.lang.exception import (
    GsTaichiSyntaxError,
)
from gstaichi.lang.matrix import MatrixType
from gstaichi.lang.struct import StructType
from gstaichi.lang.util import to_gstaichi_type
from gstaichi.types import annotations, ndarray_type, primitive_types, texture_type


class FunctionDefTransformer:
    @staticmethod
    def _decl_and_create_variable(
        ctx: ASTTransformerContext,
        annotation: Any,
        name: str,
        this_arg_features: tuple[tuple[Any, ...], ...] | None,
        invoke_later_dict: dict[str, tuple[Any, str, Callable, list[Any]]],
        prefix_name: str,
        arg_depth: int,
    ) -> tuple[bool, Any]:
        full_name = prefix_name + "_" + name
        if not isinstance(annotation, primitive_types.RefType):
            ctx.kernel_args.append(name)
        if isinstance(annotation, ArgPackType):
            assert this_arg_features is not None
            kernel_arguments.push_argpack_arg(name)
            d = {}
            items_to_put_in_dict = []
            for j, (_name, anno) in enumerate(annotation.members.items()):
                result, obj = FunctionDefTransformer._decl_and_create_variable(
                    ctx, anno, _name, this_arg_features[j], invoke_later_dict, full_name, arg_depth + 1
                )
                if not result:
                    d[_name] = None
                    items_to_put_in_dict.append((full_name + "_" + _name, _name, obj))
                else:
                    d[_name] = obj
            argpack = kernel_arguments.decl_argpack_arg(annotation, d)
            for item in items_to_put_in_dict:
                invoke_later_dict[item[0]] = argpack, item[1], *item[2]
            return True, argpack
        if annotation == annotations.template or isinstance(annotation, annotations.template):
            assert ctx.global_vars is not None
            return True, ctx.global_vars[name]
        if isinstance(annotation, annotations.sparse_matrix_builder):
            return False, (
                kernel_arguments.decl_sparse_matrix,
                (
                    to_gstaichi_type(this_arg_features),
                    full_name,
                ),
            )
        if isinstance(annotation, ndarray_type.NdarrayType):
            assert this_arg_features is not None
            return False, (
                kernel_arguments.decl_ndarray_arg,
                (
                    to_gstaichi_type(this_arg_features[0]),
                    this_arg_features[1],
                    full_name,
                    this_arg_features[2],
                    this_arg_features[3],
                ),
            )
        if isinstance(annotation, texture_type.TextureType):
            assert this_arg_features is not None
            return False, (kernel_arguments.decl_texture_arg, (this_arg_features[0], full_name))
        if isinstance(annotation, texture_type.RWTextureType):
            assert this_arg_features is not None
            return False, (
                kernel_arguments.decl_rw_texture_arg,
                (this_arg_features[0], this_arg_features[1], this_arg_features[2], full_name),
            )
        if dataclasses.is_dataclass(annotation):
            """
            So, what needs to happen now is ...
            - we'll need to expand out the nested struct too
            - eg if we have:
            @ti.kernel
            def f1(some_struct: SomeStruct):
                ...

            and SomeStruct is:
            @dataclasses.dataclass
            class SomeStruct:
                a: ti.types.NDArray[ti.32, 1]
                child: ChildStruct

            and ChildStruct is:
            @datatclasses.dataclass
            class ChildStruct:
                b: ti.types.NDArray[ti.32, 1]

            ... so we'll expand the paramters to:
            - __ti_some_struct_a: ti.types.NDArray[ti.32, 1]
            - __ti_some_struct__ti_child__ti_b: ti.types.NDArray[ti.32, 1]
            (or sometihng smilar ish)

            So... we'll loop over the fields, and send those each to
            """
        if isinstance(annotation, MatrixType):
            return True, kernel_arguments.decl_matrix_arg(annotation, name, arg_depth)
        if isinstance(annotation, StructType):
            return True, kernel_arguments.decl_struct_arg(annotation, name, arg_depth)
        return True, kernel_arguments.decl_scalar_arg(annotation, name, arg_depth)

    @staticmethod
    def _transform_kernel_arg(
        ctx: ASTTransformerContext,
        invoke_later_dict: dict[str, tuple[Any, str, Callable, list[Any]]],
        create_variable_later: dict[str, Any],
        argument_name: str,
        argument_type: Any,
        this_arg_features: tuple[Any, ...],
    ) -> None:
        if isinstance(argument_type, ArgPackType):
            kernel_arguments.push_argpack_arg(argument_name)
            d = {}
            items_to_put_in_dict: list[tuple[str, str, Any]] = []
            for j, (name, anno) in enumerate(argument_type.members.items()):
                result, obj = FunctionDefTransformer._decl_and_create_variable(
                    ctx, anno, name, this_arg_features[j], invoke_later_dict, "__argpack_" + name, 1
                )
                if not result:
                    d[name] = None
                    items_to_put_in_dict.append(("__argpack_" + name, name, obj))
                else:
                    d[name] = obj
            argpack = kernel_arguments.decl_argpack_arg(argument_type, d)
            for item in items_to_put_in_dict:
                invoke_later_dict[item[0]] = argpack, item[1], *item[2]
            create_variable_later[argument_name] = argpack
        elif dataclasses.is_dataclass(argument_type):
            ctx.create_variable(argument_name, argument_type)
            for field_idx, field in enumerate(dataclasses.fields(argument_type)):
                flat_name = f"{argument_name}__ti_{field.name}"
                if not flat_name.startswith("__ti_"):
                    flat_name = f"__ti_{flat_name}"
                # if a field is a dataclass, then feed back into process_kernel_arg recursively
                if dataclasses.is_dataclass(field.type):
                    FunctionDefTransformer._transform_kernel_arg(
                        ctx,
                        invoke_later_dict,
                        create_variable_later,
                        flat_name,
                        field.type,
                        this_arg_features[field_idx],
                    )
                else:
                    result, obj = FunctionDefTransformer._decl_and_create_variable(
                        ctx,
                        field.type,
                        flat_name,
                        this_arg_features[field_idx],
                        invoke_later_dict,
                        "",
                        0,
                    )
                    if result:
                        ctx.create_variable(flat_name, obj)
                    else:
                        decl_type_func, type_args = obj
                        obj = decl_type_func(*type_args)
                        ctx.create_variable(flat_name, obj)
        else:
            result, obj = FunctionDefTransformer._decl_and_create_variable(
                ctx,
                argument_type,
                argument_name,
                this_arg_features if ctx.arg_features is not None else None,
                invoke_later_dict,
                "",
                0,
            )
            if result:
                ctx.create_variable(argument_name, obj)
            else:
                decl_type_func, type_args = obj
                obj = decl_type_func(*type_args)
                ctx.create_variable(argument_name, obj)

    @staticmethod
    def _transform_as_kernel(ctx: ASTTransformerContext, node: ast.FunctionDef, args: ast.arguments) -> None:
        assert ctx.func is not None
        assert ctx.arg_features is not None
        if node.returns is not None:
            if not isinstance(node.returns, ast.Constant):
                assert ctx.func.return_type is not None
                for return_type in ctx.func.return_type:
                    kernel_arguments.decl_ret(return_type)
        compiling_callable = impl.get_runtime().compiling_callable
        assert compiling_callable is not None
        compiling_callable.finalize_rets()

        invoke_later_dict: dict[str, tuple[Any, str, Callable, list[Any]]] = dict()
        create_variable_later: dict[str, Any] = dict()
        for i, _ in enumerate(args.args):
            arg_meta = ctx.func.arg_metas[i]
            FunctionDefTransformer._transform_kernel_arg(
                ctx,
                invoke_later_dict,
                create_variable_later,
                arg_meta.name,
                arg_meta.annotation,
                ctx.arg_features[i] if ctx.arg_features is not None else (),
            )

        for k, v in invoke_later_dict.items():
            argpack, name, func, params = v
            argpack[name] = func(*params)
        for k, v in create_variable_later.items():
            ctx.create_variable(k, v)

        compiling_callable.finalize_params()
        # remove original args
        node.args.args = []

    @staticmethod
    def _transform_func_arg(
        ctx: ASTTransformerContext,
        argument_name: str,
        argument_type: Any,
        data: Any,
    ) -> None:
        # Template arguments are passed by reference.
        if isinstance(argument_type, annotations.template):
            ctx.create_variable(argument_name, data)
            return None

        if dataclasses.is_dataclass(argument_type):
            for field in dataclasses.fields(argument_type):
                flat_name = f"{argument_name}__ti_{field.name}"
                if not flat_name.startswith("__ti_"):
                    flat_name = f"__ti_{flat_name}"
                data_child = getattr(data, field.name)
                if isinstance(
                    data_child,
                    (
                        _ndarray.ScalarNdarray,
                        matrix.VectorNdarray,
                        matrix.MatrixNdarray,
                        any_array.AnyArray,
                    ),
                ):
                    field.type.check_matched(data_child.get_type(), field.name)
                    ctx.create_variable(flat_name, data_child)
                elif dataclasses.is_dataclass(data_child):
                    FunctionDefTransformer._transform_func_arg(
                        ctx,
                        flat_name,
                        field.type,
                        getattr(data, field.name),
                    )
                else:
                    raise GsTaichiSyntaxError(
                        f"Argument {field.name} of type {argument_type} {field.type} is not recognized."
                    )
            return None

        # Ndarray arguments are passed by reference.
        if isinstance(argument_type, (ndarray_type.NdarrayType)):
            if not isinstance(
                data,
                (
                    _ndarray.ScalarNdarray,
                    matrix.VectorNdarray,
                    matrix.MatrixNdarray,
                    any_array.AnyArray,
                ),
            ):
                raise GsTaichiSyntaxError(f"Argument {argument_name} of type {argument_type} is not recognized.")
            argument_type.check_matched(data.get_type(), argument_name)
            ctx.create_variable(argument_name, data)
            return None

        # Matrix arguments are passed by value.
        if isinstance(argument_type, (MatrixType)):
            # "data" is expected to be an Expr here,
            # so we simply call "impl.expr_init_func(data)" to perform:
            #
            # TensorType* t = alloca()
            # assign(t, data)
            #
            # We created local variable "t" - a copy of the passed-in argument "data"
            if not isinstance(data, expr.Expr) or not data.ptr.is_tensor():
                raise GsTaichiSyntaxError(
                    f"Argument {argument_name} of type {argument_type} is expected to be a Matrix, but got {type(data)}."
                )

            element_shape = data.ptr.get_rvalue_type().shape()
            if len(element_shape) != argument_type.ndim:
                raise GsTaichiSyntaxError(
                    f"Argument {argument_name} of type {argument_type} is expected to be a Matrix with ndim {argument_type.ndim}, but got {len(element_shape)}."
                )

            assert argument_type.ndim > 0
            if element_shape[0] != argument_type.n:
                raise GsTaichiSyntaxError(
                    f"Argument {argument_name} of type {argument_type} is expected to be a Matrix with n {argument_type.n}, but got {element_shape[0]}."
                )

            if argument_type.ndim == 2 and element_shape[1] != argument_type.m:
                raise GsTaichiSyntaxError(
                    f"Argument {argument_name} of type {argument_type} is expected to be a Matrix with m {argument_type.m}, but got {element_shape[0]}."
                )

            ctx.create_variable(argument_name, impl.expr_init_func(data))
            return None

        if id(argument_type) in primitive_types.type_ids:
            ctx.create_variable(argument_name, impl.expr_init_func(ti_ops.cast(data, argument_type)))
            return None
        # Create a copy for non-template arguments,
        # so that they are passed by value.
        var_name = argument_name
        ctx.create_variable(var_name, impl.expr_init_func(data))
        return None

    @staticmethod
    def _transform_as_func(ctx: ASTTransformerContext, node: ast.FunctionDef, args: ast.arguments) -> None:
        assert ctx.argument_data is not None
        assert ctx.func is not None
        for data_i, data in enumerate(ctx.argument_data):
            argument = ctx.func.arg_metas[data_i]
            FunctionDefTransformer._transform_func_arg(
                ctx,
                argument.name,
                argument.annotation,
                data,
            )

        # deal with dataclasses

        # pylint: disable=import-outside-toplevel
        from gstaichi.lang.kernel_impl import Func

        assert isinstance(ctx.func, Func)
        for v in ctx.func.orig_arguments:
            if dataclasses.is_dataclass(v.annotation):
                ctx.create_variable(v.name, v.annotation)

    @staticmethod
    def build_FunctionDef(
        ctx: ASTTransformerContext,
        node: ast.FunctionDef,
        build_stmts: Callable[[ASTTransformerContext, list[ast.stmt]], None],
    ) -> None:
        if ctx.visited_funcdef:
            raise GsTaichiSyntaxError(
                f"Function definition is not allowed in 'ti.{'kernel' if ctx.is_kernel else 'func'}'."
            )
        ctx.visited_funcdef = True

        args = node.args
        assert args.vararg is None
        assert args.kwonlyargs == []
        assert args.kw_defaults == []
        assert args.kwarg is None

        if ctx.is_kernel:  # ti.kernel
            FunctionDefTransformer._transform_as_kernel(ctx, node, args)

        else:  # ti.func
            assert ctx.argument_data is not None
            assert ctx.func is not None
            if ctx.is_real_function:
                FunctionDefTransformer._transform_as_kernel(ctx, node, args)
            else:
                FunctionDefTransformer._transform_as_func(ctx, node, args)

        with ctx.variable_scope_guard():
            build_stmts(ctx, node.body)
