#pragma once
#ifdef TI_WITH_VULKAN

#include "taichi_core_impl.h"
#include "taichi_gfx_impl.h"
#include "gs_taichi/rhi/vulkan/vulkan_loader.h"
#include "gs_taichi/rhi/vulkan/vulkan_device.h"
#include "gs_taichi/rhi/vulkan/vulkan_device_creator.h"

class VulkanRuntime;
class VulkanRuntimeImported;
class VulkanRuntimeOwned;
class VulkanContext;

class VulkanRuntime : public GfxRuntime {
 public:
  VulkanRuntime();

  gs_taichi::lang::vulkan::VulkanDevice &get_vk();
  virtual TiImage allocate_image(
      const gs_taichi::lang::ImageParams &params) override final;
  virtual void free_image(TiImage image) override final;
};
class VulkanRuntimeImported : public VulkanRuntime {
  // A dirty workaround to ensure the device is fully initialized before
  // construction of `gfx_runtime_`.
  struct Workaround {
    gs_taichi::lang::vulkan::VulkanDevice vk_device;
    Workaround(uint32_t api_version,
               const gs_taichi::lang::vulkan::VulkanDevice::Params &params);
  } inner_;
  gs_taichi::lang::gfx::GfxRuntime gfx_runtime_;

 public:
  VulkanRuntimeImported(
      uint32_t api_version,
      const gs_taichi::lang::vulkan::VulkanDevice::Params &params);

  virtual gs_taichi::lang::Device &get() override final;
  virtual gs_taichi::lang::gfx::GfxRuntime &get_gfx_runtime() override final;
};
class VulkanRuntimeOwned : public VulkanRuntime {
  gs_taichi::lang::vulkan::VulkanDeviceCreator vk_device_creator_;
  gs_taichi::lang::gfx::GfxRuntime gfx_runtime_;

 public:
  VulkanRuntimeOwned();
  VulkanRuntimeOwned(
      const gs_taichi::lang::vulkan::VulkanDeviceCreator::Params &params);

  virtual gs_taichi::lang::Device &get() override final;
  virtual gs_taichi::lang::gfx::GfxRuntime &get_gfx_runtime() override final;
};

gs_taichi::lang::vulkan::VulkanDeviceCreator::Params
make_vulkan_runtime_creator_params();

#endif  // TI_WITH_VULKAN
