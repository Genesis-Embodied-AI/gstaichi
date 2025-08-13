import dataclasses
import weakref
from typing import Any, Callable, Union

import gstaichi.lang
import gstaichi.lang._ndarray
import gstaichi.lang._texture
import gstaichi.lang.expr
import gstaichi.lang.snode
from gstaichi._lib import core as _ti_core
from gstaichi.lang import _dataclass_util
from gstaichi.lang.any_array import AnyArray
from gstaichi.lang.exception import (
    GsTaichiRuntimeTypeError,
)
from gstaichi.lang.kernel_arguments import ArgMetadata
from gstaichi.lang.matrix import MatrixType
from gstaichi.lang.util import is_ti_template, to_gstaichi_type
from gstaichi.types import (
    ndarray_type,
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

    @staticmethod
    def extract_arg(arg: Any, annotation: AnnotationType, arg_name: str) -> Any:
        if is_ti_template(annotation):
            if isinstance(arg, gstaichi.lang.snode.SNode):
                return arg.ptr
            if isinstance(arg, gstaichi.lang.expr.Expr):
                return arg.ptr.get_underlying_ptr_address()
            if isinstance(arg, _ti_core.ExprCxx):
                return arg.get_underlying_ptr_address()
            if isinstance(arg, tuple):
                return tuple(TemplateMapper.extract_arg(item, annotation, arg_name) for item in arg)
            if isinstance(arg, gstaichi.lang._ndarray.Ndarray):
                raise GsTaichiRuntimeTypeError(
                    "Ndarray shouldn't be passed in via `ti.template()`, please annotate your kernel using `ti.types.ndarray(...)` instead"
                )

            if isinstance(arg, (list, tuple, dict, set)) or hasattr(arg, "_data_oriented"):
                # [Composite arguments] Return weak reference to the object
                # GsTaichi kernel will cache the extracted arguments, thus we can't simply return the original argument.
                # Instead, a weak reference to the original value is returned to avoid memory leak.

                # TODO(zhanlue): replacing "tuple(args)" with "hash of argument values"
                # This can resolve the following issues:
                # 1. Invalid weak-ref will leave a dead(dangling) entry in both caches: "self.mapping" and "self.compiled_functions"
                # 2. Different argument instances with same type and same value, will get templatized into seperate kernels.
                return weakref.ref(arg)

            # [Primitive arguments] Return the value
            return arg
        if dataclasses.is_dataclass(annotation):
            _res_l = []
            for field in dataclasses.fields(annotation):
                field_value = getattr(arg, field.name)
                child_name = _dataclass_util.create_flat_name(arg_name, field.name)
                field_extracted = TemplateMapper.extract_arg(field_value, field.type, child_name)
                _res_l.append(field_extracted)
            return tuple(_res_l)
        if isinstance(annotation, texture_type.TextureType):
            if not isinstance(arg, gstaichi.lang._texture.Texture):
                raise GsTaichiRuntimeTypeError(f"Argument {arg_name} must be a texture, got {type(arg)}")
            if arg.num_dims != annotation.num_dimensions:
                raise GsTaichiRuntimeTypeError(
                    f"TextureType dimension mismatch for argument {arg_name}: expected {annotation.num_dimensions}, got {arg.num_dims}"
                )
            return (arg.num_dims,)
        if isinstance(annotation, texture_type.RWTextureType):
            if not isinstance(arg, gstaichi.lang._texture.Texture):
                raise GsTaichiRuntimeTypeError(f"Argument {arg_name} must be a texture, got {type(arg)}")
            if arg.num_dims != annotation.num_dimensions:
                raise GsTaichiRuntimeTypeError(
                    f"RWTextureType dimension mismatch for argument {arg_name}: expected {annotation.num_dimensions}, got {arg.num_dims}"
                )
            if arg.fmt != annotation.fmt:
                raise GsTaichiRuntimeTypeError(
                    f"RWTextureType format mismatch for argument {arg_name}: expected {annotation.fmt}, got {arg.fmt}"
                )
            # (penguinliong) '0' is the assumed LOD level. We currently don't
            # support mip-mapping.
            return arg.num_dims, arg.fmt, 0
        if isinstance(annotation, ndarray_type.NdarrayType):
            if isinstance(arg, gstaichi.lang._ndarray.Ndarray):
                annotation.check_matched(arg.get_type(), arg_name)
                needs_grad = (arg.grad is not None) if annotation.needs_grad is None else annotation.needs_grad
                assert arg.shape is not None
                return arg.element_type, len(arg.shape), needs_grad, annotation.boundary
            if isinstance(arg, AnyArray):
                ty = arg.get_type()
                annotation.check_matched(arg.get_type(), arg_name)
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
                            f"Invalid value for argument {arg_name} - required array has ndim={annotation.ndim} element_dim={annotation.dtype.ndim}, "
                            f"array with {len(shape)} dimensions is provided"
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
            needs_grad = (
                getattr(arg, "requires_grad", False) if annotation.needs_grad is None else annotation.needs_grad
            )
            element_type = (
                _ti_core.get_type_factory_instance().get_tensor_type(element_shape, dtype)
                if len(element_shape) != 0
                else arg.dtype
            )
            return element_type, len(shape) - len(element_shape), needs_grad, annotation.boundary
        if isinstance(annotation, sparse_matrix_builder):
            return arg.dtype
        # Use '#' as a placeholder because other kinds of arguments are not involved in template instantiation
        return "#"

    def extract(self, args: tuple[Any, ...]) -> tuple[Any, ...]:
        extracted: list[Any] = []
        for arg, kernel_arg in zip(args, self.arguments):
            extracted.append(self.extract_arg(arg, kernel_arg.annotation, kernel_arg.name))
        return tuple(extracted)

    def lookup(self, args: tuple[Any, ...]) -> tuple[int, tuple[Any, ...]]:
        if len(args) != self.num_args:
            raise TypeError(f"{self.num_args} argument(s) needed but {len(args)} provided.")

        key = self.extract(args)
        if key not in self.mapping:
            count = len(self.mapping)
            self.mapping[key] = count
        return self.mapping[key], key
