#include "gstaichi/struct/struct.h"

namespace gstaichi::lang {

class FakeStructCompiler : public StructCompiler {
 public:
  void generate_types(SNode &) override {
  }

  void generate_child_accessors(SNode &) override {
  }

  void run(SNode &root) override {
  }
};

}  // namespace gstaichi::lang
