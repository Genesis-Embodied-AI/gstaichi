#pragma once
#include "gstaichi/rhi/device.h"

namespace gstaichi::lang {
namespace metal {

bool is_metal_api_available();

std::shared_ptr<Device> create_metal_device();

}  // namespace metal
}  // namespace gstaichi::lang
