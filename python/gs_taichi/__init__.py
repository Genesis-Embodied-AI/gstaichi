# type: ignore

from gs_taichi import (
    ad,
    algorithms,
    experimental,
    graph,
    linalg,
    math,
    sparse,
    tools,
    types,
)
from gs_taichi._funcs import *
from gs_taichi._lib import core as _ti_core
from gs_taichi._lib.utils import warn_restricted_version
from gs_taichi._logging import *
from gs_taichi._snode import *
from gs_taichi.lang import *  # pylint: disable=W0622 # TODO(archibate): It's `taichi.lang.core` overriding `taichi.core`
from gs_taichi.types.annotations import *

# Provide a shortcut to types since they're commonly used.
from gs_taichi.types.primitive_types import *

# Issue#2223: Do not reorder, or we're busted with partially initialized module
from gs_taichi import aot  # isort:skip


def __getattr__(attr):
    if attr == "cfg":
        return None if lang.impl.get_runtime().prog is None else lang.impl.current_cfg()
    raise AttributeError(f"module '{__name__}' has no attribute '{attr}'")


__version__ = (
    _ti_core.get_version_major(),
    _ti_core.get_version_minor(),
    _ti_core.get_version_patch(),
)

del _ti_core

warn_restricted_version()
del warn_restricted_version
