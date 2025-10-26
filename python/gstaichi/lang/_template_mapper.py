import weakref
from dataclasses import _FIELD, _FIELDS
from typing import Any, Callable, Union

from gstaichi._lib import core as _ti_core
from gstaichi.lang._dataclass_util import create_flat_name
from gstaichi.lang._ndarray import Ndarray
from gstaichi.lang._texture import Texture
from gstaichi.lang.any_array import AnyArray
from gstaichi.lang.exception import GsTaichiRuntimeTypeError
from gstaichi.lang.expr import Expr
from gstaichi.lang.kernel_arguments import ArgMetadata
from gstaichi.lang.matrix import MatrixType
from gstaichi.lang.snode import SNode
from gstaichi.lang.util import to_gstaichi_type
from gstaichi.types import (
    ndarray_type,
    primitive_types,
    sparse_matrix_builder,
    template,
    texture_type,
)
from gstaichi.types.enums import AutodiffMode

CompiledKernelKeyType = tuple[Callable, int, AutodiffMode]


AnnotationType = Union[
    template,
    "texture_type.TextureType",
    "texture_type.RWTextureType",
    ndarray_type.NdarrayType,
    sparse_matrix_builder,
    Any,
]


_ExprCxx = _ti_core.ExprCxx
_composite_mutable_types = {list, dict, set}
_primitive_types = {int, float, bool}


def _extract_arg(raise_on_templated_floats: bool, arg: Any, annotation: AnnotationType, arg_name: str) -> Any:
    annotation_type = type(annotation)
    arg_type = type(arg)
    if annotation is template or annotation_type is template:
        if arg_type is SNode:
            return arg.ptr
        if arg_type is Expr:
            return arg.ptr.get_underlying_ptr_address()
        if arg_type is _ExprCxx:
            return arg.get_underlying_ptr_address()
        if issubclass(arg_type, tuple):  # Handle all tuple-based containers, incl. NamedTuple
            return tuple([_extract_arg(raise_on_templated_floats, item, annotation, arg_name) for item in arg])
        if issubclass(arg_type, Ndarray):
            raise GsTaichiRuntimeTypeError(
                "Ndarray shouldn't be passed in via `ti.template()`, please annotate your kernel using `ti.types.ndarray(...)` instead"
            )
        if arg_type in _composite_mutable_types or getattr(arg_type, "_data_oriented", False):
            # [Composite arguments] Return weak reference to the object
            # GsTaichi kernel will cache the extracted arguments, thus we can't simply return the original argument.
            # Instead, a weak reference to the original value is returned to avoid memory leak.

            # TODO(zhanlue): replacing "tuple(args)" with "hash of argument values"
            # This can resolve the following issues:
            # 1. Invalid weak-ref will leave a dead(dangling) entry in both caches: "self.mapping" and "self.compiled_functions"
            # 2. Different argument instances with same type and same value, will get templatized into seperate kernels.
            return weakref.ref(arg)

        # [Primitive arguments] Return the value
        if raise_on_templated_floats and arg_type is float:
            raise ValueError("Floats not allowed as templated types.")
        return arg
    if annotation_type is ndarray_type.NdarrayType:
        if isinstance(arg, Ndarray):
            # Allow deferring '__debug__' evaluation at runtime
            if __debug__ and __builtins__["__debug__"]:
                annotation.check_matched(arg.get_type(), arg_name)
                assert arg.shape is not None
            needs_grad = annotation.needs_grad
            if needs_grad is None:
                needs_grad = arg.grad is not None
            # Convert singleton primitive dtype to int. This will dramatically speed up hashing later on.
            type_id = id(arg.element_type)
            element_type = type_id if type_id in primitive_types.type_ids else arg.element_type
            return element_type, len(arg.shape), needs_grad, annotation.boundary
        if isinstance(arg, AnyArray):
            ty = arg.get_type()
            if __debug__ and __builtins__["__debug__"]:
                annotation.check_matched(ty, arg_name)
            return ty.element_type, len(arg.shape), ty.needs_grad, annotation.boundary
        # external arrays
        shape = getattr(arg, "shape", None)
        if shape is None:
            raise GsTaichiRuntimeTypeError(f"Invalid type for argument {arg_name}, got {arg}")
        shape = tuple(shape)
        element_shape: tuple[int, ...] = ()
        dtype = to_gstaichi_type(arg.dtype)
        if isinstance(annotation.dtype, MatrixType):
            if annotation.ndim is not None:
                if len(shape) != annotation.dtype.ndim + annotation.ndim:
                    raise ValueError(
                        f"Invalid value for argument {arg_name} - required array has ndim={annotation.ndim} "
                        f"element_dim={annotation.dtype.ndim}, array with {len(shape)} dimensions is provided"
                    )
            else:
                if len(shape) < annotation.dtype.ndim:
                    raise ValueError(
                        f"Invalid value for argument {arg_name} - required element_dim={annotation.dtype.ndim}, "
                        f"array with {len(shape)} dimensions is provided"
                    )
            element_shape = shape[-annotation.dtype.ndim :]
            anno_element_shape = annotation.dtype.get_shape()
            if None not in anno_element_shape and element_shape != anno_element_shape:
                raise ValueError(
                    f"Invalid value for argument {arg_name} - required element_shape={anno_element_shape}, "
                    f"array with element shape of {element_shape} is provided"
                )
        elif annotation.dtype is not None:
            # User specified scalar dtype
            if annotation.dtype != dtype:
                raise ValueError(
                    f"Invalid value for argument {arg_name} - required array has dtype={annotation.dtype.to_string()}, "
                    f"array with dtype={dtype.to_string()} is provided"
                )

            if annotation.ndim is not None and len(shape) != annotation.ndim:
                raise ValueError(
                    f"Invalid value for argument {arg_name} - required array has ndim={annotation.ndim}, "
                    f"array with {len(shape)} dimensions is provided"
                )
        needs_grad = getattr(arg, "requires_grad", False) if annotation.needs_grad is None else annotation.needs_grad
        if element_shape:
            element_type = _ti_core.get_type_factory_instance().get_tensor_type(element_shape, dtype)
        else:
            element_type = arg.dtype
        return element_type, len(shape) - len(element_shape), needs_grad, annotation.boundary
    annotation_fields = getattr(annotation, _FIELDS, None)
    if annotation_fields is not None:
        return tuple(
            [
                _extract_arg(
                    raise_on_templated_floats,
                    getattr(arg, field.name),
                    field.type,
                    create_flat_name(arg_name, field.name),
                )
                for field in annotation_fields.values()
                if field._field_type is _FIELD
            ]
        )
    if annotation_type is texture_type.TextureType:
        if arg_type is not Texture:
            raise GsTaichiRuntimeTypeError(f"Argument {arg_name} must be a texture, got {type(arg)}")
        if arg.num_dims != annotation.num_dimensions:
            raise GsTaichiRuntimeTypeError(
                f"TextureType dimension mismatch for argument {arg_name}: expected {annotation.num_dimensions}, "
                f"got {arg.num_dims}"
            )
        return (arg.num_dims,)
    if annotation_type is texture_type.RWTextureType:
        if arg_type is not Texture:
            raise GsTaichiRuntimeTypeError(f"Argument {arg_name} must be a texture, got {type(arg)}")
        if arg.num_dims != annotation.num_dimensions:
            raise GsTaichiRuntimeTypeError(
                f"RWTextureType dimension mismatch for argument {arg_name}: expected {annotation.num_dimensions}, "
                f"got {arg.num_dims}"
            )
        if arg.fmt != annotation.fmt:
            raise GsTaichiRuntimeTypeError(
                f"RWTextureType format mismatch for argument {arg_name}: expected {annotation.fmt}, got {arg.fmt}"
            )
        # (penguinliong) '0' is the assumed LOD level. We currently don't support mip-mapping.
        return arg.num_dims, arg.fmt, 0
    if annotation_type is sparse_matrix_builder:
        return arg.dtype
    # Use '#' as a placeholder because other kinds of arguments are not involved in template instantiation
    return "#"


class TemplateMapper:
    """
    This should probably be renamed to sometihng like FeatureMapper, or
    FeatureExtractor, since:
    - it's not specific to templates
    - it extracts what are later called 'features', for example for ndarray this includes:
        - element type
        - number dimensions
        - needs grad (or not)
    - these are returned as a heterogeneous tuple, whose contents depends on the type
    """

    def __init__(self, arguments: list[ArgMetadata], template_slot_locations: list[int]) -> None:
        self.arguments: list[ArgMetadata] = arguments
        self.num_args: int = len(arguments)
        self.template_slot_locations: list[int] = template_slot_locations
        self.mapping: dict[tuple[Any, ...], int] = {}

    def extract(self, raise_on_templated_floats: bool, args: tuple[Any, ...]) -> tuple[Any, ...]:
        return tuple(
            [
                _extract_arg(raise_on_templated_floats, arg, kernel_arg.annotation, kernel_arg.name)
                for arg, kernel_arg in zip(args, self.arguments)
            ]
        )

    def lookup(self, raise_on_templated_floats: bool, args: tuple[Any, ...]) -> tuple[int, tuple[Any, ...]]:
        if len(args) != self.num_args:
            raise TypeError(f"{self.num_args} argument(s) needed but {len(args)} provided.")

        key = self.extract(raise_on_templated_floats, args)
        try:
            return self.mapping[key], key
        except KeyError:
            count = len(self.mapping)
            self.mapping[key] = count
            return count, key
