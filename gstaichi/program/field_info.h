#pragma once
#include "gstaichi/ir/type_utils.h"
#include "gstaichi/ir/snode.h"
#include "gstaichi/rhi/device.h"
#include "gstaichi/program/program.h"

namespace gstaichi {

namespace ui {

enum class FieldSource : int {
  GsTaichiNDarray = 0,
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
  DEFINE_PROPERTY(gstaichi::lang::DataType, dtype);
  DEFINE_PROPERTY(gstaichi::lang::DeviceAllocation, dev_alloc);

  FieldInfo() {
    valid = false;
  }
};

gstaichi::lang::DevicePtr get_device_ptr(gstaichi::lang::Program *program,
                                       gstaichi::lang::SNode *snode);

}  // namespace ui

}  // namespace gstaichi
