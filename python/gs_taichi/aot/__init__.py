# type: ignore

"""Taichi's AOT (ahead of time) module.

Users can use Taichi as a GPU compute shader/kernel compiler by compiling their
Taichi kernels into an AOT module.
"""

import gs_taichi.aot.conventions
from gs_taichi.aot._export import export, export_as
from gs_taichi.aot.conventions.gfxruntime140 import GfxRuntime140
from gs_taichi.aot.module import Module
