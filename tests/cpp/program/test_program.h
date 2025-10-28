#pragma once

#include <memory>

#include "gstaichi/program/program.h"

namespace gstaichi::lang {

class TestProgram {
 public:
  void setup(Arch arch = Arch::x64);

  Program *prog() {
    return prog_.get();
  }

 private:
  std::unique_ptr<Program> prog_{nullptr};
};

}  // namespace gstaichi::lang
