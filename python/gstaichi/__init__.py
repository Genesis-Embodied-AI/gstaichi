# type: ignore
from gstaichi._lib import core as _ti_core

__version__ = (
    _ti_core.get_version_major(),
    _ti_core.get_version_minor(),
    _ti_core.get_version_patch(),
)
__version_str__ = ".".join(map(str, __version__))

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


warn_restricted_version()
del warn_restricted_version

__all__ = [
    "ad",
    "algorithms",
    "experimental",
    "linalg",
    "math",
    "sparse",
    "tools",
    "types",
]
