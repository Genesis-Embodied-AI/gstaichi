#pragma once
#include "gs_taichi/aot/graph_data.h"

namespace gs_taichi {
namespace lang {
namespace directx12 {
class KernelImpl : public aot::Kernel {
 public:
  explicit KernelImpl() {
  }

  void launch(LaunchContextBuilder &ctx) override {
  }
};
}  // namespace directx12
}  // namespace lang
}  // namespace taichi
