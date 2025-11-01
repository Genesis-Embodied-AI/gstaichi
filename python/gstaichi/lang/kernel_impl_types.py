# pre-emptively putting things here, to try to avoid import loops
# (since this is used by fastcache)

from typing import Callable
from ..types.enums import AutodiffMode

CompiledKernelKeyType = tuple[Callable, int, AutodiffMode]
