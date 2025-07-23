#pragma once
#ifdef TI_WITH_METAL
#include "taichi_core_impl.h"
#include "taichi_gfx_impl.h"
#include "gs_taichi/rhi/metal/metal_device.h"

namespace capi {

class MetalRuntime;

class MetalRuntime : public GfxRuntime {
 private:
  std::unique_ptr<gs_taichi::lang::metal::MetalDevice> mtl_device_;
  gs_taichi::lang::gfx::GfxRuntime gfx_runtime_;

 public:
  explicit MetalRuntime();
  explicit MetalRuntime(
      std::unique_ptr<gs_taichi::lang::metal::MetalDevice> &&mtl_device);

  gs_taichi::lang::Device &get() override;
  gs_taichi::lang::gfx::GfxRuntime &get_gfx_runtime() override;

  gs_taichi::lang::metal::MetalDevice &get_mtl();
  virtual TiImage allocate_image(
      const gs_taichi::lang::ImageParams &params) override final;
  virtual void free_image(TiImage image) override final;
};

}  // namespace capi

#endif  // TI_WITH_METAL
