#pragma once
#include "gs_taichi/common/core.h"
#include "gs_taichi/rhi/device.h"

#ifdef TI_WITH_DX12

#endif

namespace gs_taichi::lang {
namespace directx12 {

bool is_dx12_api_available();

std::shared_ptr<Device> make_dx12_device();

std::vector<uint8_t> validate_and_sign(
    std::vector<uint8_t> &input_dxil_container);

}  // namespace directx12
}  // namespace gs_taichi::lang
