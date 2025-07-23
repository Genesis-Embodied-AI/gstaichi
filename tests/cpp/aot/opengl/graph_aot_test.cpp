#include "gtest/gtest.h"
#include "gs_taichi/rhi/opengl/opengl_api.h"
#include "tests/cpp/aot/gfx_utils.h"

using namespace gs_taichi;
using namespace lang;

TEST(CGraphAotTest, OpenglRunCGraph1) {
  if (!opengl::is_opengl_api_available()) {
    return;
  }

  auto device = gs_taichi::lang::opengl::make_opengl_device();
  aot_test_utils::run_cgraph1(Arch::opengl, device.get());
}

TEST(CGraphAotTest, OpenglRunCGraph2) {
  if (!opengl::is_opengl_api_available()) {
    return;
  }

  auto device = gs_taichi::lang::opengl::make_opengl_device();
  aot_test_utils::run_cgraph2(Arch::opengl, device.get());
}

TEST(CGraphAotTest, OpenglMpm88) {
  if (!opengl::is_opengl_api_available()) {
    return;
  }

  auto device = gs_taichi::lang::opengl::make_opengl_device();
  aot_test_utils::run_mpm88_graph(Arch::opengl, device.get());
}
