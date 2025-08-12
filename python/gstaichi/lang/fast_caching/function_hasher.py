import os
from typing import TYPE_CHECKING, Iterable

from gstaichi.lang._wrap_inspect import FunctionSourceInfo

from .fast_caching_types import HashedFunctionSourceInfo
from .hash_utils import hash_string

if TYPE_CHECKING:
    from gstaichi.lang.kernel_impl import GsTaichiCallable


def pure(fn: "GsTaichiCallable") -> "GsTaichiCallable":
    fn.is_pure = True
    return fn


def _hash_function(function_info: FunctionSourceInfo) -> str:
    with open(function_info.filepath) as f:
        contents = f.read().split("\n")
    lines = contents[function_info.start_lineno : function_info.end_lineno]
    _hash = hash_string("\n".join(lines))
    return _hash


def hash_functions(function_infos: Iterable[FunctionSourceInfo]) -> list[HashedFunctionSourceInfo]:
    results = []
    for f_info in function_infos:
        _hash = _hash_function(f_info)
        results.append(HashedFunctionSourceInfo(function_source_info=f_info, hash=_hash))
    return results


def hash_kernel(kernel_info: FunctionSourceInfo) -> str:
    return _hash_function(kernel_info)


def dump_stats() -> None:
    print("function hasher dump stats")


def _validate_hashed_function_info(hashed_function_info: HashedFunctionSourceInfo) -> bool:
    """
    Checks the hash
    """
    if not os.path.isfile(hashed_function_info.function_source_info.filepath):
        return False
    _hash = _hash_function(hashed_function_info.function_source_info)
    return _hash == hashed_function_info.hash


def validate_hashed_function_infos(function_infos: Iterable[HashedFunctionSourceInfo]) -> bool:
    for function_info in function_infos:
        if not _validate_hashed_function_info(function_info):
            return False
    return True
