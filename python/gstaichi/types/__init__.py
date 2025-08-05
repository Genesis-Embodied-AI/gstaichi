# type: ignore

"""
This module defines data types in GsTaichi:

- primitive: int, float, etc.
- compound: matrix, vector, struct.
- template: for reference types.
- ndarray: for arbitrary arrays.
- quant: for quantized types, see "https://yuanming.gstaichi.graphics/publication/2021-quangstaichi/quangstaichi.pdf"
"""

from gstaichi.types import quant
from gstaichi.types.annotations import *
from gstaichi.types.compound_types import *
from gstaichi.types.ndarray_type import *
from gstaichi.types.primitive_types import *
from gstaichi.types.texture_type import *
from gstaichi.types.utils import *
