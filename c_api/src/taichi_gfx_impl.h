#pragma once

#include "taichi_core_impl.h"
#include "gs_taichi/runtime/gfx/runtime.h"
#include "gs_taichi/common/virtual_dir.h"

class GfxRuntime;

class GfxRuntime : public Runtime {
 public:
  GfxRuntime(gs_taichi::Arch arch);
  virtual gs_taichi::lang::gfx::GfxRuntime &get_gfx_runtime() = 0;

  virtual Error create_aot_module(const gs_taichi::io::VirtualDir *dir,
                                  TiAotModule &out) override final;
  virtual void buffer_copy(const gs_taichi::lang::DevicePtr &dst,
                           const gs_taichi::lang::DevicePtr &src,
                           size_t size) override final;
  virtual void copy_image(
      const gs_taichi::lang::DeviceAllocation &dst,
      const gs_taichi::lang::DeviceAllocation &src,
      const gs_taichi::lang::ImageCopyParams &params) override final;
  virtual void track_image(const gs_taichi::lang::DeviceAllocation &image,
                           gs_taichi::lang::ImageLayout layout) override final;
  virtual void untrack_image(
      const gs_taichi::lang::DeviceAllocation &image) override final;
  virtual void transition_image(
      const gs_taichi::lang::DeviceAllocation &image,
      gs_taichi::lang::ImageLayout layout) override final;
  virtual void flush() override final;
  virtual void wait() override final;
};
