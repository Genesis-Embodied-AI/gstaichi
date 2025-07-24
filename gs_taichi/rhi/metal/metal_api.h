#pragma once
#include "gs_taichi/rhi/device.h"

namespace gs_taichi::lang {
namespace metal {

bool is_metal_api_available();

std::shared_ptr<Device> create_metal_device();

}  // namespace metal
}  // namespace gs_taichi::lang
