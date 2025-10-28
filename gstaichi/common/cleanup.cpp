#include "gstaichi/common/cleanup.h"
#include <utility>

namespace gstaichi {

RaiiCleanup::RaiiCleanup(Func fn) : fn_(std::move(fn)) {
}

RaiiCleanup::~RaiiCleanup() {
  if (fn_) {
    fn_();
    fn_ = nullptr;
  }
}

RaiiCleanup make_cleanup(RaiiCleanup::Func fn) {
  return RaiiCleanup{std::move(fn)};
}

}  // namespace gstaichi
