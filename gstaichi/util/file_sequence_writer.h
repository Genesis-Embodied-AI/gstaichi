#pragma once

#include "gstaichi/util/lang_util.h"
#ifdef TI_WITH_LLVM
#include "gstaichi/runtime/llvm/llvm_fwd.h"
#endif

namespace gstaichi::lang {
class IRNode;
}  // namespace gstaichi::lang

namespace gstaichi {

class FileSequenceWriter {
 public:
  FileSequenceWriter(std::string filename_template, std::string file_type);

#ifdef TI_WITH_LLVM
  // returns filename
  std::string write(llvm::Module *module);
#endif

  std::string write(lang::IRNode *irnode);

  std::string write(const std::string &str);

 private:
  int counter_;
  std::string filename_template_;
  std::string file_type_;

  std::pair<std::ofstream, std::string> create_new_file();
};

}  // namespace gstaichi
