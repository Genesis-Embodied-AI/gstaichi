#include "gtest/gtest.h"

#include "gs_taichi/common/core.h"

namespace gs_taichi {

TEST(CoreTest, Basic) {
  EXPECT_EQ(trim_string("hello taichi  "), "hello taichi");
}

}  // namespace taichi
