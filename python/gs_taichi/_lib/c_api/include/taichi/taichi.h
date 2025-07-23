#pragma once
#ifndef TAICHI_H
#define TAICHI_H

#include "gs_taichi_platform.h"

#include "gs_taichi_core.h"

#ifdef TI_WITH_VULKAN
#include "gs_taichi_vulkan.h"
#endif  // TI_WITH_VULKAN

#ifdef TI_WITH_OPENGL
#include "gs_taichi_opengl.h"
#endif  // TI_WITH_OPENGL

#ifdef TI_WITH_CUDA
#include "gs_taichi_cuda.h"
#endif  // TI_WITH_CUDA

#ifdef TI_WITH_CPU
#include "gs_taichi_cpu.h"
#endif  // TI_WITH_CPU

#ifdef TI_WITH_METAL
#include "gs_taichi_metal.h"
#endif  // TI_WITH_METAL

#endif  // TAICHI_H
