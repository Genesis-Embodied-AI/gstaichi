import dataclasses
import enum
import time
from typing import Any, Sequence

import numpy as np

from gstaichi.lang._ndarray import ScalarNdarray
from gstaichi.lang.fast_caching import FIELD_METADATA_CACHE_VALUE
from gstaichi.lang.field import ScalarField
from gstaichi.lang.matrix import MatrixField, MatrixNdarray, VectorNdarray
from gstaichi.lang.util import is_data_oriented

from .hash_utils import hash_string

g_num_calls = 0
g_num_args = 0
g_hashing_time = 0
g_repr_time = 0
g_num_ignored_calls = 0


def dataclass_to_repr(arg: Any) -> str:
    repr_l = []
    for field in dataclasses.fields(arg):
        child_value = getattr(arg, field.name)
        _repr = to_representative_str(child_value)
        full_repr = f"{field.name}: ({_repr})"
        if field.metadata.get(FIELD_METADATA_CACHE_VALUE, False):
            full_repr += f" = {child_value}"
        repr_l.append(full_repr)
    return "[" + ",".join(repr_l) + "]"


def to_representative_str(arg: Any) -> str | None:
    """
    string should somehow represent the type of arg. Doesnt have to be hashed, nor does it have
    to be the actual python type string, just something representative of the type, and won't collide
    with different (allowed) types
    """
    arg_type = type(arg)
    if isinstance(arg, ScalarNdarray):
        return f"[nd-{arg.dtype}-{len(arg.shape)}]"
    if isinstance(arg, VectorNdarray):
        return f"[ndv-{arg.n}-{arg.dtype}-{len(arg.shape)}]"
    if isinstance(arg, ScalarField):
        return f"[f-{arg.snode._id}-{arg.dtype}-{arg.shape}]"
    if isinstance(arg, MatrixNdarray):
        return f"[ndm-{arg.m}-{arg.n}-{arg.dtype}-{len(arg.shape)}]"
    if "torch.Tensor" in str(arg_type):
        return f"[pt-{arg.dtype}-{arg.ndim}]"
    if isinstance(arg, np.ndarray):
        return f"[np-{arg.dtype}-{arg.ndim}]"
    if isinstance(arg, MatrixField):
        return f"[fm-{arg.m}-{arg.n}-{arg.snode._id}-{arg.dtype}-{arg.shape}]"
    if dataclasses.is_dataclass(arg):
        return dataclass_to_repr(arg)
    if is_data_oriented(arg):
        child_repr_l = []
        for k, v in arg.__dict__.items():
            _child_repr = to_representative_str(v)
            if _child_repr is None:
                print("not representable child", k, type(v))
                return None
            child_repr_l.append(f"{k}: {_child_repr}")
        return ", ".join(child_repr_l)
    if arg_type in [int, float, np.float32, np.float64, np.int32, np.int64, bool, np.bool]:
        return str(arg_type)
    if isinstance(arg, enum.Enum):
        return f"enum-{arg.name}-{arg.value}"
    return None


def hash_args(args: Sequence[Any]) -> str | None:
    global g_num_calls, g_num_args, g_hashing_time, g_repr_time, g_num_ignored_calls
    g_num_calls += 1
    g_num_args += len(args)
    hash_l = []
    for arg in args:
        start = time.time()
        _hash = to_representative_str(arg)
        g_repr_time += time.time() - start
        if not _hash:
            g_num_ignored_calls += 1
            return None
        hash_l.append(_hash)
    start = time.time()
    res = hash_string("_".join(hash_l))
    g_hashing_time += time.time() - start
    return res


def dump_stats() -> None:
    print("args hasher dump stats")
    print("total calls", g_num_calls)
    print("ignored calls", g_num_ignored_calls)
    print("total args", g_num_args)
    print("hashing time", g_hashing_time)
    print("arg representation time", g_repr_time)
