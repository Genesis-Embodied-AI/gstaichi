#include "gstaichi/runtime/llvm/kernel_launcher.h"
#include "gstaichi/codegen/ir_dump.h"
#include "gstaichi/util/file_sequence_writer.h"

namespace gstaichi::lang {
namespace LLVM {

KernelLauncher::KernelLauncher(Config config) : config_(std::move(config)) {
}

void KernelLauncher::launch_kernel(
    const lang::CompiledKernelData &compiled_kernel_data,
    LaunchContextBuilder &ctx) {
  std::cout << "LLVM::KernelLauncher::launch_kerne 1" << std::endl;
  TI_ASSERT(arch_uses_llvm(compiled_kernel_data.arch()));
  const auto &llvm_ckd =
      dynamic_cast<const LLVM::CompiledKernelData &>(compiled_kernel_data);
  std::cout << "LLVM::KernelLauncher::launch_kernel" << std::endl;

  // Dump LLVM IR to file
  const auto& tasks = llvm_ckd.get_internal_data().compiled_data.tasks;
  std::string kernel_name = "unknown_kernel";
  if (!tasks.empty()) {
      // Task names are formatted as "{kernel_name}_{task_id}_{task_type}{suffix}"
      // Extract the kernel name (everything before the first underscore followed by a digit)
      std::string task_name = tasks[0].name;
      size_t pos = task_name.find_first_of("_0123456789");
      if (pos != std::string::npos) {
          kernel_name = task_name.substr(0, pos);
      } else {
          kernel_name = task_name;  // fallback to full task name
      }
  }
  std::string filename = "/tmp/ir/kernel_" + kernel_name + "_" + std::to_string(launch_id_counter_) + "_llvm_before_launch.ll";
  std::error_code EC;
  llvm::raw_fd_ostream dest_file(filename, EC);
  if (!EC) {
      llvm_ckd.get_internal_data().compiled_data.module->print(dest_file, nullptr);
      std::cout << "LLVM IR dumped to: " << filename << std::endl;
  } else {
      std::cout << "Failed to dump LLVM IR: " << EC.message() << std::endl;
  }

  auto handle = register_llvm_kernel(llvm_ckd);
  launch_llvm_kernel(handle, ctx);

}

}  // namespace LLVM
}  // namespace gstaichi::lang
