#include "gtest/gtest.h"

#include "gstaichi/common/core.h"

namespace gstaichi {

TEST(CoreTest, Basic) {
  EXPECT_EQ(trim_string("hello gstaichi  "), "hello gstaichi");
}

}  // namespace gstaichi
