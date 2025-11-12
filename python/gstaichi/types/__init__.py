"""
This module defines data types in GsTaichi:

- primitive: int, float, etc.
- compound: matrix, vector, struct.
- template: for reference types.
- ndarray: for arbitrary arrays.
- quant: for quantized types, see "https://yuanming.gstaichi.graphics/publication/2021-quangstaichi/quangstaichi.pdf"
"""

from gstaichi.types import quant
from gstaichi.types.annotations import *  # type: ignore
from gstaichi.types.compound_types import *  # type: ignore
from gstaichi.types.ndarray_type import *  # type: ignore
from gstaichi.types.primitive_types import *  # type: ignore
from gstaichi.types.utils import *  # type: ignore

__all__ = ["quant"]
