#include "gtest/gtest.h"
#include "gs_taichi/rhi/vulkan/vulkan_device.h"
#include "gs_taichi/rhi/vulkan/vulkan_device_creator.h"
#include "gs_taichi/rhi/vulkan/vulkan_loader.h"
#include "tests/cpp/aot/gfx_utils.h"

using namespace gs_taichi;
using namespace lang;

TEST(CGraphAotTest, VulkanRunCGraph2) {
  // Otherwise will segfault on macOS VM,
  // where Vulkan is installed but no devices are present
  if (!vulkan::is_vulkan_api_available()) {
    return;
  }

  // Create Taichi Device for computation
  lang::vulkan::VulkanDeviceCreator::Params evd_params;
  evd_params.api_version = std::nullopt;
  auto embedded_device =
      std::make_unique<gs_taichi::lang::vulkan::VulkanDeviceCreator>(evd_params);
  gs_taichi::lang::vulkan::VulkanDevice *device_ =
      static_cast<gs_taichi::lang::vulkan::VulkanDevice *>(
          embedded_device->device());
  aot_test_utils::run_cgraph2(Arch::vulkan, device_);
}

TEST(CGraphAotTest, VulkanRunCGraph1) {
  // Otherwise will segfault on macOS VM,
  // where Vulkan is installed but no devices are present
  if (!vulkan::is_vulkan_api_available()) {
    return;
  }

  // Create Taichi Device for computation
  lang::vulkan::VulkanDeviceCreator::Params evd_params;
  evd_params.api_version = std::nullopt;
  auto embedded_device =
      std::make_unique<gs_taichi::lang::vulkan::VulkanDeviceCreator>(evd_params);
  gs_taichi::lang::vulkan::VulkanDevice *device_ =
      static_cast<gs_taichi::lang::vulkan::VulkanDevice *>(
          embedded_device->device());
  aot_test_utils::run_cgraph1(Arch::vulkan, device_);
}

TEST(CGraphAotTest, VulkanMpm88) {
  // Otherwise will segfault on macOS VM,
  // where Vulkan is installed but no devices are present
  if (!vulkan::is_vulkan_api_available()) {
    return;
  }

  // Create Taichi Device for computation
  lang::vulkan::VulkanDeviceCreator::Params evd_params;
  evd_params.api_version = std::nullopt;
  auto embedded_device =
      std::make_unique<gs_taichi::lang::vulkan::VulkanDeviceCreator>(evd_params);
  gs_taichi::lang::vulkan::VulkanDevice *device_ =
      static_cast<gs_taichi::lang::vulkan::VulkanDevice *>(
          embedded_device->device());
  aot_test_utils::run_mpm88_graph(Arch::vulkan, device_);
}
