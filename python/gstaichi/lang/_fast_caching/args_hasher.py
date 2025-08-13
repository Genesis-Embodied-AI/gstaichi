import dataclasses
import enum
import time
from typing import Any, Sequence

import numpy as np

from .._ndarray import ScalarNdarray
from ..field import ScalarField
from ..matrix import MatrixField, MatrixNdarray, VectorNdarray
from ..util import is_data_oriented
from . import FIELD_METADATA_CACHE_VALUE
from .hash_utils import hash_iterable_strings

g_num_calls = 0
g_num_args = 0
g_hashing_time = 0
g_repr_time = 0
g_num_ignored_calls = 0


def dataclass_to_repr(arg: Any) -> str:
    repr_l = []
    for field in dataclasses.fields(arg):
        child_value = getattr(arg, field.name)
        _repr = stringify_obj_type(child_value)
        full_repr = f"{field.name}: ({_repr})"
        if field.metadata.get(FIELD_METADATA_CACHE_VALUE, False):
            full_repr += f" = {child_value}"
        repr_l.append(full_repr)
    return "[" + ",".join(repr_l) + "]"


def stringify_obj_type(obj: Any) -> str | None:
    """
    stringify the type of obj

    String should somehow represent the type of obj. Doesnt have to be hashed, nor does it have
    to be the actual python type string, just something representative of the type, and won't collide
    with different (allowed) types.
    """
    arg_type = type(obj)
    if isinstance(obj, ScalarNdarray):
        return f"[nd-{obj.dtype}-{len(obj.shape)}]"
    if isinstance(obj, VectorNdarray):
        return f"[ndv-{obj.n}-{obj.dtype}-{len(obj.shape)}]"
    if isinstance(obj, ScalarField):
        return f"[f-{obj.snode._id}-{obj.dtype}-{obj.shape}]"
    if isinstance(obj, MatrixNdarray):
        return f"[ndm-{obj.m}-{obj.n}-{obj.dtype}-{len(obj.shape)}]"
    if "torch.Tensor" in str(arg_type):
        return f"[pt-{obj.dtype}-{obj.ndim}]"
    if isinstance(obj, np.ndarray):
        return f"[np-{obj.dtype}-{obj.ndim}]"
    if isinstance(obj, MatrixField):
        return f"[fm-{obj.m}-{obj.n}-{obj.snode._id}-{obj.dtype}-{obj.shape}]"
    if dataclasses.is_dataclass(obj):
        return dataclass_to_repr(obj)
    if is_data_oriented(obj):
        child_repr_l = []
        for k, v in obj.__dict__.items():
            _child_repr = stringify_obj_type(v)
            if _child_repr is None:
                print("not representable child", k, type(v))
                return None
            child_repr_l.append(f"{k}: {_child_repr}")
        return ", ".join(child_repr_l)
    if arg_type in [int, float, np.float32, np.float64, np.int32, np.int64, bool, np.bool]:
        return str(arg_type)
    if isinstance(obj, enum.Enum):
        return f"enum-{obj.name}-{obj.value}"
    return None


def hash_args(args: Sequence[Any]) -> str | None:
    global g_num_calls, g_num_args, g_hashing_time, g_repr_time, g_num_ignored_calls
    g_num_calls += 1
    g_num_args += len(args)
    hash_l = []
    for arg in args:
        start = time.time()
        _hash = stringify_obj_type(arg)
        g_repr_time += time.time() - start
        if not _hash:
            g_num_ignored_calls += 1
            return None
        hash_l.append(_hash)
    start = time.time()
    res = hash_iterable_strings(hash_l)
    g_hashing_time += time.time() - start
    return res


def dump_stats() -> None:
    print("args hasher dump stats")
    print("total calls", g_num_calls)
    print("ignored calls", g_num_ignored_calls)
    print("total args", g_num_args)
    print("hashing time", g_hashing_time)
    print("arg representation time", g_repr_time)
