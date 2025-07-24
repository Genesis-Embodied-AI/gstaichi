#pragma once
#ifdef TI_WITH_OPENGL

#include "taichi_gfx_impl.h"
#include "gs_taichi/rhi/opengl/opengl_api.h"
#include "gs_taichi/rhi/opengl/opengl_device.h"

class OpenglRuntime : public GfxRuntime {
 private:
  gs_taichi::lang::opengl::GLDevice device_;
  gs_taichi::lang::gfx::GfxRuntime gfx_runtime_;

 public:
  OpenglRuntime();
  virtual gs_taichi::lang::Device &get() override final;
  virtual gs_taichi::lang::gfx::GfxRuntime &get_gfx_runtime() override final;
  gs_taichi::lang::opengl::GLDevice &get_gl() {
    return device_;
  }
};

#endif  // TI_WITH_OPENGL
