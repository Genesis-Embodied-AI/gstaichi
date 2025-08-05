#pragma once

#include "gstaichi/ir/type.h"

namespace gstaichi::lang {

/**
 * Matrix wrapper used in AOT.
 */
class Matrix {
 public:
  explicit Matrix(const uint32_t &length,
                  const DataType dtype,
                  const intptr_t &data)
      : length_(length), dtype_(dtype), data_(data) {
  }

  DataType dtype() const {
    return dtype_;
  }

  uint32_t length() const {
    // number of matrix elements
    return length_;
  }

  intptr_t data() const {
    return data_;
  }

 private:
  uint32_t length_{};
  DataType dtype_{};
  intptr_t data_{};
};

}  // namespace gstaichi::lang
