#pragma once
#include "gs_taichi/ir/type_utils.h"
#include "gs_taichi/ir/snode.h"
#include "gs_taichi/rhi/device.h"
#include "gs_taichi/program/program.h"

namespace gs_taichi {

namespace ui {

enum class FieldSource : int {
  TaichiNDarray = 0,
  HostMappedPtr = 1,
};

#define DEFINE_PROPERTY(Type, name)       \
  Type name;                              \
  void set_##name(const Type &new_name) { \
    name = new_name;                      \
  }                                       \
  Type get_##name() {                     \
    return name;                          \
  }

struct FieldInfo {
  DEFINE_PROPERTY(bool, valid)
  DEFINE_PROPERTY(std::vector<int>, shape);
  DEFINE_PROPERTY(uint64_t, num_elements);
  DEFINE_PROPERTY(FieldSource, field_source);
  DEFINE_PROPERTY(gs_taichi::lang::DataType, dtype);
  DEFINE_PROPERTY(gs_taichi::lang::DeviceAllocation, dev_alloc);

  FieldInfo() {
    valid = false;
  }
};

gs_taichi::lang::DevicePtr get_device_ptr(gs_taichi::lang::Program *program,
                                       gs_taichi::lang::SNode *snode);

}  // namespace ui

}  // namespace taichi
