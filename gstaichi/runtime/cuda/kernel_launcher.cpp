#include "gstaichi/runtime/cuda/kernel_launcher.h"
#include "gstaichi/rhi/cuda/cuda_context.h"

namespace gstaichi::lang {
namespace cuda {

static std::unordered_map<size_t, char *> device_arg_buffer_by_hash;
static std::unordered_map<size_t, size_t> arg_buffer_size_by_hash;

class ArgsManager {
 public:
  ArgsManager() {
  }
  void prepare_args(
      LaunchContextBuilder &ctx,
      gstaichi::lang::cuda::KernelLauncherContext &launcher_ctx,
      LlvmRuntimeExecutor *executor,
      const std::vector<std::pair<std::vector<int>, CallableBase::Parameter>>
          &parameters,
      char const *const device_result_buffer) {
    size_t parameters_hash = hash_parameters_by_address(parameters);
    std::cout << "parameters hash " << parameters_hash << std::endl;
    auto device_arg_buffer_it = device_arg_buffer_by_hash.find(parameters_hash);
    if (device_arg_buffer_it != device_arg_buffer_by_hash.end()) {
      char *device_arg_buffer = device_arg_buffer_it->second;
      ctx.arg_buffer_size = arg_buffer_size_by_hash[parameters_hash];
      ctx.get_context().arg_buffer = device_arg_buffer;
      std::cout << " cache hit device arg buffer " << (void *)device_arg_buffer << " arg buffer size " << ctx.arg_buffer_size << std::endl;
      return;
    }

    for (int i = 0; i < (int)parameters.size(); i++) {
      const auto &kv = parameters[i];
      const auto &key = kv.first;
      const auto &parameter = kv.second;
      if (parameter.is_array) {
        const auto arr_sz = ctx.array_runtime_sizes[key];
        // Note: both numpy and PyTorch support arrays/tensors with zeros
        // in shapes, e.g., shape=(0) or shape=(100, 0, 200). This makes
        // `arr_sz` zero.
        if (arr_sz == 0) {
          continue;
        }

        std::vector<int> data_ptr_idx = key;
        data_ptr_idx.push_back(TypeFactory::DATA_PTR_POS_IN_NDARRAY);
        auto data_ptr = ctx.array_ptrs[data_ptr_idx];
        std::vector<int> grad_ptr_idx = key;
        grad_ptr_idx.push_back(TypeFactory::GRAD_PTR_POS_IN_NDARRAY);
        auto grad_ptr = ctx.array_ptrs[grad_ptr_idx];
        if (ctx.device_allocation_type[key] ==
            LaunchContextBuilder::DevAllocType::kNone) {
          // External array
          // Note: assuming both data & grad are on the same device
          if (on_cuda_device(data_ptr)) {
            // data_ptr is a raw ptr on CUDA device
            device_ptrs[data_ptr_idx] = data_ptr;
            device_ptrs[grad_ptr_idx] = grad_ptr;
          } else {
            DeviceAllocation devalloc = executor->allocate_memory_on_device(
                arr_sz, (uint64 *)device_result_buffer);
            device_ptrs[data_ptr_idx] =
                executor->get_device_alloc_info_ptr(devalloc);
            transfers[data_ptr_idx] = {data_ptr, devalloc};


            CUDADriver::get_instance().memcpy_host_to_device(
                (void *)device_ptrs[data_ptr_idx], data_ptr, arr_sz);
            if (grad_ptr != nullptr) {
              DeviceAllocation grad_devalloc =
                  executor->allocate_memory_on_device(
                      arr_sz, (uint64 *)device_result_buffer);
              device_ptrs[grad_ptr_idx] =
                  executor->get_device_alloc_info_ptr(grad_devalloc);
              transfers[grad_ptr_idx] = {grad_ptr, grad_devalloc};

              CUDADriver::get_instance().memcpy_host_to_device(
                  (void *)device_ptrs[grad_ptr_idx], grad_ptr, arr_sz);
            } else {
              device_ptrs[grad_ptr_idx] = nullptr;
            }
          }

          ctx.set_ndarray_ptrs(key, (uint64)device_ptrs[data_ptr_idx],
                               (uint64)device_ptrs[grad_ptr_idx]);
        } else if (arr_sz > 0) {
          // Ndarray
          DeviceAllocation *ptr = static_cast<DeviceAllocation *>(data_ptr);
          // Unwrapped raw ptr on device
          device_ptrs[data_ptr_idx] = executor->get_device_alloc_info_ptr(*ptr);

          if (grad_ptr != nullptr) {
            ptr = static_cast<DeviceAllocation *>(grad_ptr);
            device_ptrs[grad_ptr_idx] =
                executor->get_device_alloc_info_ptr(*ptr);
          } else {
            device_ptrs[grad_ptr_idx] = nullptr;
          }

          ctx.set_ndarray_ptrs(key, (uint64)device_ptrs[data_ptr_idx],
                               (uint64)device_ptrs[grad_ptr_idx]);
        }
      }
    }
    if (transfers.size() > 0) {
      CUDADriver::get_instance().stream_synchronize(nullptr);
    }
    if (ctx.arg_buffer_size > 0) {
      CUDADriver::get_instance().malloc_async((void **)&device_arg_buffer,
                                              ctx.arg_buffer_size, nullptr);
      CUDADriver::get_instance().memcpy_host_to_device_async(
          device_arg_buffer, ctx.get_context().arg_buffer, ctx.arg_buffer_size,
          nullptr);
      ctx.get_context().arg_buffer = device_arg_buffer;
    }
    if (transfers.size() == 0) {
      device_arg_buffer_by_hash[parameters_hash] = device_arg_buffer;
      arg_buffer_size_by_hash[parameters_hash] = ctx.arg_buffer_size;
      std::cout << "caching for parameters hash " << parameters_hash << " device arg buffer " << (void *)device_arg_buffer << " arg buffer size " << ctx.arg_buffer_size << std::endl;
      caching_arg_buffer = true;
    }
  }

  void after_call(LaunchContextBuilder &ctx, LlvmRuntimeExecutor *executor) {
    if (caching_arg_buffer) {
      return;
      // shouldnt free
      // and transfers is zero because we made caching conditional on it being
      // so
    }
    if (ctx.arg_buffer_size > 0) {
      CUDADriver::get_instance().mem_free_async(device_arg_buffer, nullptr);
    }
    // copy data back to host
    if (transfers.size() > 0) {
      CUDADriver::get_instance().stream_synchronize(nullptr);
      for (auto itr = transfers.begin(); itr != transfers.end(); itr++) {
        auto &idx = itr->first;
        auto arg_id = idx;
        arg_id.pop_back();
        CUDADriver::get_instance().memcpy_device_to_host(
            itr->second.first, (void *)device_ptrs[idx],
            ctx.array_runtime_sizes[arg_id]);
        executor->deallocate_memory_on_device(itr->second.second);
      }
    }
  }

 private:
  // |transfers| is only used for external arrays whose data is originally on
  // host. They are first transferred onto device and that device pointer is
  // stored in |device_ptrs| below. |transfers| saves its original pointer so
  // that we can copy the data back once kernel finishes. as well as the
  // temporary device allocations, which can be freed after kernel finishes. Key
  // is [arg_id, ptr_pos], where ptr_pos is TypeFactory::DATA_PTR_POS_IN_NDARRAY
  // for data_ptr and TypeFactory::GRAD_PTR_POS_IN_NDARRAY for grad_ptr. Value
  // is [host_ptr, temporary_device_alloc]. Invariant: temp_devallocs.size() !=
  // 0 <==> transfer happened.
  std::unordered_map<std::vector<int>,
                     std::pair<void *, DeviceAllocation>,
                     hashing::Hasher<std::vector<int>>>
      transfers;

  // |device_ptrs| stores pointers on device for all arrays args, including
  // external arrays and ndarrays, no matter whether the data is originally on
  // device or host.
  // This is the source of truth for us to look for device pointers used in CUDA
  // kernels.
  std::
      unordered_map<std::vector<int>, void *, hashing::Hasher<std::vector<int>>>
          device_ptrs;
  char *device_arg_buffer = nullptr;
  bool caching_arg_buffer = false;

  bool on_cuda_device(void *ptr) {
    unsigned int attr_val = 0;
    uint32_t ret_code = CUDADriver::get_instance().mem_get_attribute.call(
        &attr_val, CU_POINTER_ATTRIBUTE_MEMORY_TYPE, (void *)ptr);

    return ret_code == CUDA_SUCCESS && attr_val == CU_MEMORYTYPE_DEVICE;
  }

  size_t hash_parameters_by_address(
      const std::vector<std::pair<std::vector<int>, CallableBase::Parameter>>
          &parameters) {
    size_t seed = 0;
    std::cout << "hashing " << parameters.size() << " parameters by address" << std::endl;
    for (const auto &kv : parameters) {
      // Hash the address of kv.second
      auto addr = reinterpret_cast<uintptr_t>(&kv.second);
      std::cout << "  - kv.second " << kv.second << " &kv.second " << (void *)addr << std::endl;
      seed ^=
          std::hash<uintptr_t>{}(addr) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }
    return seed;
  }
};

class ResultBufferManager {
 public:
  void prepare_result_buffer(LaunchContextBuilder &ctx,
                             LlvmRuntimeExecutor *executor) {
    CUDADriver::get_instance().malloc_async(
        (void **)&device_result_buffer,
        std::max(ctx.result_buffer_size, sizeof(uint64)), nullptr);
    ctx.get_context().runtime = executor->get_llvm_runtime();

    host_result_buffer = (char *)ctx.get_context().result_buffer;
    if (ctx.result_buffer_size > 0) {
      ctx.get_context().result_buffer = (uint64 *)device_result_buffer;
    }
  }
  char const *const get_device_result_buffer() const {
    return device_result_buffer;
  }
  void after_call(LaunchContextBuilder &ctx) {
    if (ctx.result_buffer_size > 0) {
      CUDADriver::get_instance().memcpy_device_to_host_async(
          host_result_buffer, device_result_buffer, ctx.result_buffer_size,
          nullptr);
    }
    CUDADriver::get_instance().mem_free_async(device_result_buffer, nullptr);
  }

 private:
  char *host_result_buffer{nullptr};
  char *device_result_buffer{nullptr};
};

void KernelLauncher::launch_llvm_kernel(Handle handle,
                                        LaunchContextBuilder &ctx) {
  TI_ASSERT(handle.get_launch_id() < contexts_.size());
  KernelLauncherContext launcher_ctx = contexts_[handle.get_launch_id()];
  LlvmRuntimeExecutor *executor = get_runtime_executor();
  JITModule *cuda_module = launcher_ctx.jit_module;
  const std::vector<std::pair<std::vector<int>, CallableBase::Parameter>>
      &parameters = launcher_ctx.parameters;
  const std::vector<OffloadedTask> &offloaded_tasks =
      launcher_ctx.offloaded_tasks;

  CUDAContext::get_instance().make_current();

  ResultBufferManager result_buffer_manager;
  result_buffer_manager.prepare_result_buffer(ctx, executor);

  ArgsManager args_manager;
  args_manager.prepare_args(ctx, launcher_ctx, executor, parameters,
                            result_buffer_manager.get_device_result_buffer());

  for (auto task : offloaded_tasks) {
    TI_TRACE("Launching kernel {}<<<{}, {}>>>", task.name, task.grid_dim,
             task.block_dim);
    cuda_module->launch(task.name, task.grid_dim, task.block_dim, 0,
                        {&ctx.get_context()}, {});
  }
  args_manager.after_call(ctx, executor);
  result_buffer_manager.after_call(ctx);
}

KernelLauncher::Handle KernelLauncher::register_llvm_kernel(
    const LLVM::CompiledKernelData &compiled) {
  TI_ASSERT(compiled.arch() == Arch::cuda);

  if (!compiled.get_handle()) {
    auto handle = make_handle();
    auto index = handle.get_launch_id();
    contexts_.resize(index + 1);

    auto &ctx = contexts_[index];
    auto *executor = get_runtime_executor();

    auto data = compiled.get_internal_data().compiled_data.clone();
    auto parameters = compiled.get_internal_data().args;
    auto *jit_module = executor->create_jit_module(std::move(data.module));

    // Populate ctx
    ctx.jit_module = jit_module;
    ctx.parameters = std::move(parameters);
    ctx.offloaded_tasks = std::move(data.tasks);

    compiled.set_handle(handle);
  }
  return *compiled.get_handle();
}

}  // namespace cuda
}  // namespace gstaichi::lang
