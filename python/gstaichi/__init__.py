# type: ignore

from gstaichi import (
    ad,
    algorithms,
    experimental,
    linalg,
    math,
    sparse,
    tools,
    types,
)
from gstaichi._funcs import *
from gstaichi._lib import core as _ti_core
from gstaichi._lib.utils import warn_restricted_version
from gstaichi._logging import *
from gstaichi._snode import *
from gstaichi.lang import *  # pylint: disable=W0622 # TODO(archibate): It's `gstaichi.lang.core` overriding `gstaichi.core`
from gstaichi.types.annotations import *

# Provide a shortcut to types since they're commonly used.
from gstaichi.types.primitive_types import *


def __getattr__(attr):
    if attr == "cfg":
        return None if lang.impl.get_runtime()._prog is None else lang.impl.current_cfg()
    raise AttributeError(f"module '{__name__}' has no attribute '{attr}'")


__version__ = (
    _ti_core.get_version_major(),
    _ti_core.get_version_minor(),
    _ti_core.get_version_patch(),
)

del _ti_core

warn_restricted_version()
del warn_restricted_version
