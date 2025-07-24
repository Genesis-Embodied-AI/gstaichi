#pragma once
#include "gs_taichi/common/core.h"
#include "gs_taichi/rhi/device.h"

#ifdef TI_WITH_DX11
#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "d3dcompiler.lib")
#pragma comment(lib, "dxguid.lib")
#include <d3d11.h>
#endif

namespace gs_taichi::lang {
namespace directx11 {

bool is_dx_api_available();

std::shared_ptr<Device> make_dx11_device();

}  // namespace directx11
}  // namespace gs_taichi::lang
