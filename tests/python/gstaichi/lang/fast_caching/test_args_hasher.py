import dataclasses

import numpy as np

import gstaichi as ti
from gstaichi.lang._fast_caching import FIELD_METADATA_CACHE_VALUE, args_hasher

from tests import test_utils


@test_utils.test()
def test_args_hasher_numeric() -> None:
    seen = set()
    for arg in (3, 5.3, np.int32(3), np.int64(5), np.float32(2), np.float64(2)):
        for it in (0, 1):
            hash = args_hasher.hash_args([arg])
            assert hash is not None
            if it == 0:
                assert hash not in seen
                seen.add(hash)
            else:
                assert hash in seen


@test_utils.test()
def test_args_hasher_unsupported_data_oriented() -> None:
    @ti.data_oriented
    class Foo: ...

    foo = Foo()
    assert args_hasher.hash_args([foo]) is None


@test_utils.test()
def test_args_hasher_ndarray() -> None:
    seen = set()
    for dtype in [ti.i32, ti.i64, ti.f32, ti.f64]:
        for ndim in [0, 1, 2]:
            arg = ti.ndarray(dtype, [1] * ndim)
            for it in [0, 1]:
                hash = args_hasher.hash_args([arg])
                assert hash is not None
                if it == 0:
                    assert hash not in seen
                    seen.add(hash)
                else:
                    assert hash in seen


@test_utils.test()
def test_args_hasher_ndarray_vector() -> None:
    seen = set()
    for dtype in [ti.i32, ti.i64, ti.f32, ti.f64]:
        for vector_size in [2, 3]:
            for ndim in [0, 1, 2]:
                arg = ti.Vector.ndarray(vector_size, dtype, [1] * ndim)
                for it in [0, 1]:
                    hash = args_hasher.hash_args([arg])
                    assert hash is not None
                    if it == 0:
                        assert hash not in seen
                        seen.add(hash)
                    else:
                        assert hash in seen


@test_utils.test()
def test_args_hasher_ndarray_matrix() -> None:
    seen = set()
    for dtype in [ti.i32, ti.i64, ti.f32, ti.f64]:
        for m in [2, 3]:
            for n in [2, 3]:
                for ndim in [0, 1, 2]:
                    arg = ti.Matrix.ndarray(m, n, dtype, [1] * ndim)
                    for it in [0, 1]:
                        hash = args_hasher.hash_args([arg])
                        assert hash is not None
                        if it == 0:
                            assert hash not in seen
                            seen.add(hash)
                        else:
                            assert hash in seen


def _ti_init_same_arch() -> None:
    assert ti.cfg is not None
    ti.init(arch=getattr(ti, ti.cfg.arch.name))


@test_utils.test()
def test_args_hasher_field() -> None:
    """
    This is trickier than the others, since we need to take into account snode id, and we need to reinitialize
    taichi each attempt.

    Reminder: fields have an snode id, that is assigned at creation, and is assigned sequentially, from
    the time of ti.init. If you recreate the same field, without calling ti.init in between you'll get
    a new snode id. Two fields with a different snode id should be considered different, in the context of
    deciding whether a kernel needs to be recompiled. A kernel is bound to the snode id of each kernel
    used in that kernel. No matter whether the field was accessed as a global variable, or passed in as
    a parameter. In addition the kernel is bound to the dtype and shape of the field, as well as pretty
    much everything else about the field other than the actual values/data stored in it.

    We need to check therefore that:
    - if we reinitialize ti, and recreate the same field, that the hashes DO match
    - if we do NOT reinitialize ti, and we recreate the same field, that the hashes do NOT match (because:
      different snode id)
    """
    seen = set()
    for dtype in [ti.i32, ti.i64, ti.f32, ti.f64]:
        for shape in [(2,), (5,), (2, 5)]:
            for _it in range(3):
                if _it == 0:
                    _ti_init_same_arch()
                    arg = ti.field(dtype, shape)
                    hash = args_hasher.hash_args([arg])
                    assert hash is not None
                    assert hash not in seen
                    seen.add(hash)
                elif _it == 1:
                    _ti_init_same_arch()
                    arg = ti.field(dtype, shape)
                    hash = args_hasher.hash_args([arg])
                    assert hash is not None
                    assert hash in seen
                else:
                    arg = ti.field(dtype, shape)
                    hash = args_hasher.hash_args([arg])
                    assert hash not in seen
                    seen.add(hash)


@test_utils.test()
def test_args_hasher_field_vector() -> None:
    seen = set()
    for dtype in [ti.i32, ti.i64, ti.f32, ti.f64]:
        for n in [2, 3]:
            for shape in [(2,), (5,), (2, 5)]:
                for _it in range(3):
                    if _it == 0:
                        _ti_init_same_arch()
                        arg = ti.Vector.field(n, dtype, shape)
                        hash = args_hasher.hash_args([arg])
                        assert hash is not None
                        assert hash not in seen
                        seen.add(hash)
                    elif _it == 1:
                        _ti_init_same_arch()
                        arg = ti.Vector.field(n, dtype, shape)
                        hash = args_hasher.hash_args([arg])
                        assert hash is not None
                        assert hash in seen
                    else:
                        arg = ti.Vector.field(n, dtype, shape)
                        hash = args_hasher.hash_args([arg])
                        assert hash not in seen
                        seen.add(hash)


@test_utils.test()
def test_args_hasher_field_matrix() -> None:
    seen = set()
    for dtype in [ti.i32, ti.i64, ti.f32, ti.f64]:
        for m in [2, 3]:
            for n in [2, 3]:
                for shape in [(2,), (5,), (2, 5)]:
                    for _it in range(3):
                        if _it == 0:
                            _ti_init_same_arch()
                            arg = ti.Matrix.field(m, n, dtype, shape)
                            hash = args_hasher.hash_args([arg])
                            assert hash is not None
                            assert hash not in seen
                            seen.add(hash)
                        elif _it == 1:
                            _ti_init_same_arch()
                            arg = ti.Matrix.field(m, n, dtype, shape)
                            hash = args_hasher.hash_args([arg])
                            assert hash is not None
                            assert hash in seen
                        else:
                            arg = ti.Matrix.field(m, n, dtype, shape)
                            hash = args_hasher.hash_args([arg])
                            assert hash not in seen
                            seen.add(hash)


@test_utils.test()
def test_args_hasher_field_vs_ndarray() -> None:
    a_ndarray = ti.ndarray(ti.i32, 1)
    a_field = ti.field(ti.i32, 1)
    ndarray_hash = args_hasher.hash_args([a_ndarray])
    field_hash = args_hasher.hash_args([a_field])
    assert ndarray_hash is not None
    assert field_hash is not None
    assert ndarray_hash != field_hash


@test_utils.test()
def test_cache_values_unchecked() -> None:
    """
    Should we consider two dataclasses with same fields but different name as different?
    Considering them to be the same makes testing easier for now...
    """

    @dataclasses.dataclass
    class MyConfigNoChecked:
        some_int_unchecked: int

    @dataclasses.dataclass
    class MyConfigNoCheckedSame:
        some_int_unchecked: int

    @dataclasses.dataclass
    class MyConfigNoCheckedDiff:
        some_int_new: int

    base = MyConfigNoChecked(some_int_unchecked=3)
    same = MyConfigNoCheckedSame(some_int_unchecked=6)
    diff = MyConfigNoCheckedDiff(some_int_new=3)

    h = args_hasher.hash_args
    h_base = h([base])
    assert h_base is not None
    assert h_base == h([same])
    assert h_base != h([diff])


@test_utils.test()
def test_cache_values_checked() -> None:
    @dataclasses.dataclass
    class MyConfigChecked:
        some_int_checked: int = dataclasses.field(metadata={FIELD_METADATA_CACHE_VALUE: True})

    base = MyConfigChecked(some_int_checked=5)
    same = MyConfigChecked(some_int_checked=5)
    diff = MyConfigChecked(some_int_checked=7)

    h = args_hasher.hash_args
    h_base = h([base])
    assert h_base is not None
    assert h_base == h([same])
    assert h_base != h([diff])
