# type: ignore

"""
This module defines data types in Taichi:

- primitive: int, float, etc.
- compound: matrix, vector, struct.
- template: for reference types.
- ndarray: for arbitrary arrays.
- quant: for quantized types, see "https://yuanming.taichi.graphics/publication/2021-quantaichi/quantaichi.pdf"
"""

from gs_taichi.types import quant
from gs_taichi.types.annotations import *
from gs_taichi.types.compound_types import *
from gs_taichi.types.ndarray_type import *
from gs_taichi.types.primitive_types import *
from gs_taichi.types.texture_type import *
from gs_taichi.types.utils import *
