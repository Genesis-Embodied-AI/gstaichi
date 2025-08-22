from typing import Any, Iterable, Sequence

from pydantic import BaseModel

from gstaichi import _logging

from .._wrap_inspect import FunctionSourceInfo
from . import args_hasher, config_hasher, function_hasher
from .fast_caching_types import HashedFunctionSourceInfo
from .hash_utils import hash_iterable_strings
from .python_side_cache import PythonSideCache


def create_cache_key(kernel_source_info: FunctionSourceInfo, args: Sequence[Any]) -> str | None:
    """
    cache key takes into account:
    - arg types
    - cache value arg values
    - kernel function (but not sub functions)
    - compilation config (which includes arch, and debug)
    """
    args_hash = args_hasher.hash_args(args)
    if args_hash is None:
        # the bit in caps at start should not be modified without modifying corresponding text
        # freetext bit can be freely modified
        _logging.warn(
            f"[FASTCACHE][INVALID_FUNC] The pure function {kernel_source_info.function_name} could not be "
            "fast cached, because one or more parameter types were invalid"
        )
        return None
    kernel_hash = function_hasher.hash_kernel(kernel_source_info)
    config_hash = config_hasher.hash_compile_config()
    cache_key = hash_iterable_strings((kernel_hash, args_hash, config_hash))
    return cache_key


class CacheValue(BaseModel):
    hashed_function_source_infos: list[HashedFunctionSourceInfo]


def store(cache_key: str, function_source_infos: Iterable[FunctionSourceInfo]) -> None:
    """
    Note that unlike other caches, this cache is not going to store the actual value we want.
    This cache is only used for verification that our cache key is valid. Big picture:
    - we have a cache key, based on args and top level kernel function
    - we want to use this to look up LLVM IR, in C++ side cache
    - however, before doing that, we first want to validate that the source code didn't change
        - i.e. is our cache key still valid?
    - the python side cache contains information we will use to verify that our cache key is valid
        - ie the list of function source infos
    """
    if not cache_key:
        return
    cache = PythonSideCache()
    hashed_function_source_infos = function_hasher.hash_functions(function_source_infos)
    cache_value_obj = CacheValue(hashed_function_source_infos=list(hashed_function_source_infos))
    cache.store(cache_key, cache_value_obj.json())


def _try_load(cache_key: str) -> Sequence[HashedFunctionSourceInfo] | None:
    cache = PythonSideCache()
    maybe_cache_value_json = cache.try_load(cache_key)
    if maybe_cache_value_json is None:
        return None
    cache_value_obj = CacheValue.parse_raw(maybe_cache_value_json)
    return cache_value_obj.hashed_function_source_infos


def validate_cache_key(cache_key: str) -> bool:
    """
    loads function source infos from cache, if available
    checks the hashes against the current source code
    """
    maybe_hashed_function_source_infos = _try_load(cache_key)
    if not maybe_hashed_function_source_infos:
        return False
    return function_hasher.validate_hashed_function_infos(maybe_hashed_function_source_infos)


def dump_stats() -> None:
    print("dump stats")
    args_hasher.dump_stats()
    function_hasher.dump_stats()
