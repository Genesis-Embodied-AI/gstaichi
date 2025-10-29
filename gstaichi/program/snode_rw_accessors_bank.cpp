#include "gstaichi/program/snode_rw_accessors_bank.h"

#include "gstaichi/program/program.h"

namespace gstaichi::lang {

namespace {
void set_kernel_args(const std::vector<int> &I,
                     int num_active_indices,
                     LaunchContextBuilder *launch_ctx) {
  for (int i = 0; i < num_active_indices; i++) {
    launch_ctx->set_arg_int({i}, I[i]);
  }
}
}  // namespace

SNodeRwAccessorsBank::Accessors SNodeRwAccessorsBank::get(SNode *snode) {
  auto &kernels = snode_to_kernels_[snode];
  if (kernels.reader == nullptr) {
    kernels.reader = &(program_->get_snode_reader(snode));
  }
  if (kernels.writer == nullptr) {
    kernels.writer = &(program_->get_snode_writer(snode));
  }
  return Accessors(snode, kernels, program_);
}

SNodeRwAccessorsBank::Accessors::Accessors(const SNode *snode,
                                           const RwKernels &kernels,
                                           Program *prog)
    : snode_(snode),
      prog_(prog),
      reader_(kernels.reader),
      writer_(kernels.writer) {
  TI_ASSERT(reader_ != nullptr);
  TI_ASSERT(writer_ != nullptr);
}
void SNodeRwAccessorsBank::Accessors::write_float(const std::vector<int> &I,
                                                  float64 val) {
  auto launch_ctx = writer_->make_launch_context();
  set_kernel_args(I, snode_->num_active_indices, &launch_ctx);
  launch_ctx.set_arg_float({snode_->num_active_indices}, val);
  prog_->synchronize();
  CompileResult compile_result = prog_->compile_kernel(
      prog_->compile_config(), prog_->get_device_caps(), *writer_);
  auto &compiled_kernel_data = compile_result.compiled_kernel_data;
  prog_->launch_kernel(compiled_kernel_data, launch_ctx);
}

float64 SNodeRwAccessorsBank::Accessors::read_float(const std::vector<int> &I) {
  prog_->synchronize();
  auto launch_ctx = reader_->make_launch_context();
  set_kernel_args(I, snode_->num_active_indices, &launch_ctx);
  const CompileResult compile_result = prog_->compile_kernel(
      prog_->compile_config(), prog_->get_device_caps(), *reader_);
  auto &compiled_kernel_data = compile_result.compiled_kernel_data;
  prog_->launch_kernel(compiled_kernel_data, launch_ctx);
  prog_->synchronize();
  return launch_ctx.get_struct_ret_float({0});
}

// for int32 and int64
void SNodeRwAccessorsBank::Accessors::write_int(const std::vector<int> &I,
                                                int64 val) {
  std::cout << "SNodeRwAccessorsBank::Accessors::write_int called" << " val " << val << std::endl;
  auto launch_ctx = writer_->make_launch_context();
  std::cout << "SNodeRwAccessorsBank::Accessors::write_int set_kernel_args " << " num active indices " << snode_->num_active_indices << std::endl;
  set_kernel_args(I, snode_->num_active_indices, &launch_ctx);
  std::cout << "SNodeRwAccessorsBank::Accessors::write_int launch_ctx.set_arg_int " << " num active indices " << snode_->num_active_indices << " val " << val << std::endl;
  launch_ctx.set_arg_int({snode_->num_active_indices}, val);
  std::cout << "SNodeRwAccessorsBank::Accessors::write_int after set arg int" << std::endl;
  prog_->synchronize();
  CompileResult compile_result = prog_->compile_kernel(
      prog_->compile_config(), prog_->get_device_caps(), *writer_);
  auto &compiled_kernel_data = compile_result.compiled_kernel_data;
  std::cout << "SNodeRwAccessorsBank::Accessors::write_int launch kernel" << std::endl;
  prog_->launch_kernel(compiled_kernel_data, launch_ctx);
  std::cout << "SNodeRwAccessorsBank::Accessors::write_int after launch kernel" << std::endl;
}

// for int32 and int64
void SNodeRwAccessorsBank::Accessors::write_uint(const std::vector<int> &I,
                                                 uint64 val) {
  std::cout << "SNodeRwAccessorsBank::Accessors::write_uint called" << " val " << val << std::endl;
  auto launch_ctx = writer_->make_launch_context();
  std::cout << "SNodeRwAccessorsBank::Accessors::write_uint set_kernel_args " << " num active indices " << snode_->num_active_indices << std::endl;
  set_kernel_args(I, snode_->num_active_indices, &launch_ctx);
  std::cout << "SNodeRwAccessorsBank::Accessors::write_uint launch_ctx.set_arg_int " << " num active indices " << snode_->num_active_indices << " val " << val << std::endl;
  launch_ctx.set_arg_uint({snode_->num_active_indices}, val);
  std::cout << "SNodeRwAccessorsBank::Accessors::write_uint after set arg int" << std::endl;
  prog_->synchronize();
  CompileResult compile_result = prog_->compile_kernel(
      prog_->compile_config(), prog_->get_device_caps(), *writer_);
  auto &compiled_kernel_data = compile_result.compiled_kernel_data;
  std::cout << "SNodeRwAccessorsBank::Accessors::write_uint launch kernel" << std::endl;
  prog_->launch_kernel(compiled_kernel_data, launch_ctx);
  std::cout << "SNodeRwAccessorsBank::Accessors::write_uint after launch kernel" << std::endl;
}

int64 SNodeRwAccessorsBank::Accessors::read_int(const std::vector<int> &I) {
  std::cout << "SNodeRwAccessorsBank::Accessors::read_int called" << std::endl;
  prog_->synchronize();
  auto launch_ctx = reader_->make_launch_context();
  std::cout << "SNodeRwAccessorsBank::Accessors::read_int set_kernel_args " << " num active indices " << snode_->num_active_indices << std::endl;
  set_kernel_args(I, snode_->num_active_indices, &launch_ctx);
  CompileResult compile_result = prog_->compile_kernel(
      prog_->compile_config(), prog_->get_device_caps(), *reader_);
  auto &compiled_kernel_data = compile_result.compiled_kernel_data;
  std::cout << "SNodeRwAccessorsBank::Accessors::read_int launch kernel" << std::endl;
  prog_->launch_kernel(compiled_kernel_data, launch_ctx);
  std::cout << "SNodeRwAccessorsBank::Accessors::read_int after launch kernel" << std::endl;
  prog_->synchronize();
  std::cout << "SNodeRwAccessorsBank::Accessors::read_int call get_struct_ret_int" << std::endl;
  auto res = launch_ctx.get_struct_ret_int({0});
  std::cout << "SNodeRwAccessorsBank::Accessors::read_int after call get_struct_ret_int" << std::endl;
  return res;
}

uint64 SNodeRwAccessorsBank::Accessors::read_uint(const std::vector<int> &I) {
  prog_->synchronize();
  auto launch_ctx = reader_->make_launch_context();
  set_kernel_args(I, snode_->num_active_indices, &launch_ctx);
  CompileResult compile_result = prog_->compile_kernel(
      prog_->compile_config(), prog_->get_device_caps(), *reader_);
  auto &compiled_kernel_data = compile_result.compiled_kernel_data;
  prog_->launch_kernel(compiled_kernel_data, launch_ctx);
  prog_->synchronize();
  return launch_ctx.get_struct_ret_uint({0});
}

}  // namespace gstaichi::lang
