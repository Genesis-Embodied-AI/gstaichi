import inspect
import types
import typing
from dataclasses import is_dataclass

from gstaichi.lang._template_mapper import TemplateMapper
from gstaichi.lang.exception import GsTaichiSyntaxError
from gstaichi.types import (
    ndarray_type,
    primitive_types,
    sparse_matrix_builder,
    template,
)
from gstaichi.lang.kernel_arguments import ArgMetadata
from gstaichi.lang.matrix import MatrixType
from gstaichi.lang.struct import StructType
from gstaichi.types import (
    ndarray_type,
    primitive_types,
    template,
)


class FuncBase:
    """
    Base class for Kernels and Funcs
    """
    def __init__(self, func, is_kernel: bool, is_classkernel: bool, is_classfunc: bool) -> None:
        self.func = func
        self.is_kernel = is_kernel
        # TODO: merge is_classkernel and is_classfunc?
        self.is_classkernel = is_classkernel
        self.is_classfunc = is_classfunc
        self.arg_metas: list[ArgMetadata] = []
        self.arg_metas_expanded: list[ArgMetadata] = []
        self.orig_arguments: list[ArgMetadata] = []
        self.return_type = None

        self.check_parameter_annotations()

        self.mapper = TemplateMapper(self.arg_metas, self.template_slot_locations)

    def check_parameter_annotations(self) -> None:
        """
        Look at annotations of function parameters, and store into self.arg_metas
        and self.orig_arguments (both are identical after this call)
        - they just contain the original parameter annotations for now, unexpanded
        - this function mostly just does checking
        """
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
            if self.is_kernel and param.default is not inspect.Parameter.empty:
                raise GsTaichiSyntaxError("GsTaichi kernels do not support default values for arguments")
            if param.kind == inspect.Parameter.KEYWORD_ONLY:
                raise GsTaichiSyntaxError("GsTaichi kernels do not support keyword parameters")
            if param.kind != inspect.Parameter.POSITIONAL_OR_KEYWORD:
                raise GsTaichiSyntaxError('GsTaichi kernels only support "positional or keyword" parameters')
            annotation = param.annotation
            if param.annotation is inspect.Parameter.empty:
                if i == 0 and (self.is_classkernel or self.is_classfunc):  # The |self| parameter
                    annotation = template()
                elif self.is_kernel:
                    raise GsTaichiSyntaxError("GsTaichi kernels parameters must be type annotated")
            else:
                annotation_type = type(annotation)
                if annotation_type is ndarray_type.NdarrayType:
                    pass
                elif annotation is ndarray_type.NdarrayType:
                    # convert from ti.types.NDArray into ti.types.NDArray()
                    annotation = annotation()
                elif id(annotation) in primitive_types.type_ids:
                    pass
                elif issubclass(annotation_type, MatrixType):
                    pass
                elif not self.is_kernel and annotation_type is primitive_types.RefType:
                    pass
                elif annotation_type is StructType:
                    pass
                elif annotation_type is template or annotation is template:
                    pass
                elif annotation_type is type and is_dataclass(annotation):
                    pass
                elif self.is_kernel and isinstance(annotation, sparse_matrix_builder):
                    pass
                else:
                    raise GsTaichiSyntaxError(f"Invalid type annotation (argument {i}) of Taichi kernel: {annotation}")
            self.arg_metas.append(ArgMetadata(annotation, param.name, param.default))
            self.orig_arguments.append(ArgMetadata(annotation, param.name, param.default))

        self.template_slot_locations: list[int] = []
        for i, arg in enumerate(self.arg_metas):
            if arg.annotation == template or isinstance(arg.annotation, template):
                self.template_slot_locations.append(i)
