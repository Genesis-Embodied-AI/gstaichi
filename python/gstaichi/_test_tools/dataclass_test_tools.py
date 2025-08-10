import dataclasses
from typing import Any, cast

import gstaichi as ti


def build_struct(struct_type: Any) -> Any:
    member_objects = {}
    for field in dataclasses.fields(struct_type):
        if isinstance(field.type, ti.types.NDArray):
            print("got ndarray")
            ndarray_type = cast(ti.types.ndarray, field.type)
            assert ndarray_type.ndim is not None
            shape = tuple([10] * ndarray_type.ndim)
            child_obj = ti.ndarray(ndarray_type.dtype, shape=shape)
        elif dataclasses.is_dataclass(field.type):
            child_obj = build_struct(field.type)
        elif isinstance(field.type, ti.Template) or field.type == ti.Template:
            child_obj = ti.field(ti.i32, (10,))
        else:
            raise Exception("unknown type ", field.type)
        member_objects[field.name] = child_obj
    # DataclassClass = dataclasses.make_dataclass(struct_name, declaration_type_by_name)
    dataclass_object = struct_type(**member_objects)
    return dataclass_object


def build_obj_tuple_from_type_dict(name_to_type: dict[str, Any]) -> tuple[Any, ...]:
    obj_l = []
    for name, param_type in name_to_type.items():
        if isinstance(param_type, ti.types.NDArray):
            print("got ndarray")
            ndarray_type = cast(ti.types.ndarray, param_type)
            assert ndarray_type.ndim is not None
            shape = tuple([10] * ndarray_type.ndim)
            child_obj = ti.ndarray(ndarray_type.dtype, shape=shape)
        elif dataclasses.is_dataclass(param_type):
            child_obj = build_struct(param_type)
        elif isinstance(param_type, ti.Template) or param_type == ti.Template:
            child_obj = ti.field(ti.i32, (10,))
        else:
            raise Exception("unknown type ", param_type)
        obj_l.append(child_obj)
    return tuple(obj_l)
