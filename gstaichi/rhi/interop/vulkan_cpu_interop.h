#pragma once

#include "gstaichi/rhi/device.h"

namespace gstaichi::lang {

void memcpy_cpu_to_vulkan(DevicePtr dst, DevicePtr src, uint64_t size);

void memcpy_cpu_to_vulkan_via_staging(DevicePtr dst,
                                      DevicePtr staging,
                                      DevicePtr src,
                                      uint64_t size);

}  // namespace gstaichi::lang
