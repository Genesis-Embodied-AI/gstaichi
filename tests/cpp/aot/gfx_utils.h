#pragma once
#include "gtest/gtest.h"

#include "gs_taichi/rhi/device.h"
#include "gs_taichi/aot/graph_data.h"
#include "gs_taichi/program/graph_builder.h"
#include "gs_taichi/program/program.h"

namespace gs_taichi::lang {
namespace aot_test_utils {
[[maybe_unused]] static void write_devalloc(
    gs_taichi::lang::DeviceAllocation &alloc,
    const void *data,
    size_t size);

[[maybe_unused]] static void
load_devalloc(gs_taichi::lang::DeviceAllocation &alloc, void *data, size_t size);

void view_devalloc_as_ndarray(Device *device_);

[[maybe_unused]] void run_cgraph1(Arch arch, gs_taichi::lang::Device *device_);

[[maybe_unused]] void run_cgraph2(Arch arch, gs_taichi::lang::Device *device_);

[[maybe_unused]] void run_kernel_test1(Arch arch, gs_taichi::lang::Device *device);

[[maybe_unused]] void run_kernel_test2(Arch arch, gs_taichi::lang::Device *device);

[[maybe_unused]] void run_dense_field_kernel(Arch arch,
                                             gs_taichi::lang::Device *device);

[[maybe_unused]] void run_mpm88_graph(Arch arch, gs_taichi::lang::Device *device_);
}  // namespace aot_test_utils
}  // namespace taichi::lang
