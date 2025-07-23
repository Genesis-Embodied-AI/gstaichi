#pragma once

#include <memory>

#include "gs_taichi/program/program.h"

namespace gs_taichi::lang {

class TestProgram {
 public:
  void setup(Arch arch = Arch::x64);

  Program *prog() {
    return prog_.get();
  }

 private:
  std::unique_ptr<Program> prog_{nullptr};
};

}  // namespace taichi::lang
