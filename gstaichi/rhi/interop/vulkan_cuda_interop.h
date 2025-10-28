#pragma once

#include "gstaichi/rhi/device.h"

namespace gstaichi::lang {

void memcpy_cuda_to_vulkan(DevicePtr dst, DevicePtr src, uint64_t size);

void memcpy_vulkan_to_cuda(DevicePtr dst, DevicePtr src, uint64_t size);

}  // namespace gstaichi::lang
