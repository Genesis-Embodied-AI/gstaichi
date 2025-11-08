#pragma once
#include <gstaichi/program/callable.h>
#include "gstaichi/program/ndarray.h"
#include "gstaichi/program/texture.h"
#include "gstaichi/program/matrix.h"

namespace gstaichi::lang {

struct RuntimeContext;

class LaunchContextBuilder {
 public:
  enum class DevAllocType : int8_t {
    kNone = 0,
    kNdarray = 1,
    kTexture = 2,
    kRWTexture = 3
    // kArgPack = 4,
  };

  explicit LaunchContextBuilder(CallableBase *kernel);

  LaunchContextBuilder(LaunchContextBuilder &&) = default;
  LaunchContextBuilder &operator=(LaunchContextBuilder &&) = default;
  LaunchContextBuilder(const LaunchContextBuilder &) = delete;
  LaunchContextBuilder &operator=(const LaunchContextBuilder &) = delete;

  // Copy all the arguments already added to an existing launcher context.
  // The input context must be associated with the exactly same kernel, and
  // the current context must be fresh new, without any variable already added.
  // This method is useful to speed up calling repeatedly a given kernel with
  // the exact same input arguments.
  void copy(const LaunchContextBuilder &other);

  void set_arg_float(const std::vector<int> &arg_id, float64 d);
  // Bulk processing of multiple scalar float arguments at the same time.
  // This is mainly useful to mitigate pybind11 function call overhead.
  // In this context, 'args_id' is a vector gathering the position of each
  // of these scalar arguments in the corresponding kernel. As a result, the
  // length 'args_id' and 'vec' must be equal.
  void set_args_float(const std::vector<int> &args_id,
                      const std::vector<float64> &vec);

  // Created signed and unsigned version for argument range check of pybind
  void set_arg_int(const std::vector<int> &arg_id, int64 d);
  // Bulk processing of multiple scalar int arguments at the same time.
  // See 'set_arg_float' documentation for details.
  void set_args_int(const std::vector<int> &args_id,
                    const std::vector<int64> &vec);
  // Bulk processing of multiple scalar uint arguments at the same time.
  // See 'set_arg_float' documentation for details.
  void set_arg_uint(const std::vector<int> &arg_id, uint64 d);
  void set_args_uint(const std::vector<int> &args_id,
                     const std::vector<uint64> &vec);

  void set_array_runtime_size(const std::vector<int> &i, uint64 size);

  void set_array_device_allocation_type(const std::vector<int> &i,
                                        DevAllocType usage);

  template <typename T>
  void set_arg(const std::vector<int> &i, T v);

  // The following two functions can be used to set struct args and primitive
  // args. The first element of `arg_indices` is the index of the argument. The
  // rest of the elements are the index of the field in each depth of the nested
  // struct.

  template <typename T>
  void set_struct_arg_impl(std::vector<int> arg_indices, T v);

  template <typename T>
  void set_struct_arg(std::vector<int> arg_indices, T v);

  void set_ndarray_ptrs(const std::vector<int> &arg_id,
                        uint64 data_ptr,
                        uint64 grad_ptr);

  template <typename T>
  T get_arg(const std::vector<int> &i);

  template <typename T>
  T get_struct_arg(std::vector<int> arg_indices);

  template <typename T>
  T get_ret(int i);

  void set_arg_external_array_with_shape(const std::vector<int> &arg_id,
                                         uintptr_t ptr,
                                         uint64 size,
                                         const std::vector<int64> &shape,
                                         uintptr_t grad_ptr = 0);

  void set_arg_ndarray_impl(const std::vector<int> &arg_id,
                            intptr_t devalloc_ptr,
                            const std::vector<int> &shape,
                            intptr_t devalloc_ptr_grad = 0);
  void set_arg_ndarray(const std::vector<int> &arg_id, const Ndarray &arr);
  // Bulk processing of multiple individual Taichi NDarray arguments (without
  // any associated gradient) at the same time.
  // See 'set_arg_float' for details.
  void set_args_ndarray(const std::vector<int> &args_id,
                        const std::vector<Ndarray *> &arrs);
  void set_arg_ndarray_with_grad(const std::vector<int> &arg_id,
                                 const Ndarray &arr,
                                 const Ndarray &arr_grad);
  // Bulk processing of multiple individual Taichi NDarray arguments (along
  // with associated gradient) at the same time.
  // See 'set_arg_float' for details.
  void set_args_ndarray_with_grad(const std::vector<int> &args_id,
                                  const std::vector<Ndarray *> &arrs,
                                  const std::vector<Ndarray *> &arrs_grad);

  void set_arg_matrix(int arg_id, const Matrix &matrix);

  TypedConstant fetch_ret(const std::vector<int> &index);
  float64 get_struct_ret_float(const std::vector<int> &index);
  int64 get_struct_ret_int(const std::vector<int> &index);
  uint64 get_struct_ret_uint(const std::vector<int> &index);

  RuntimeContext &get_context();

 private:
  TypedConstant fetch_ret_impl(int offset, const Type *dt);
  CallableBase *kernel_;
  std::unique_ptr<RuntimeContext> owned_ctx_;
  // |ctx_| *almost* always points to |owned_ctx_|. However, it is possible
  // that the caller passes a RuntimeContext pointer externally. In that case,
  // |owned_ctx_| will be nullptr.
  // Invariant: |ctx_| will never be nullptr.
  RuntimeContext *ctx_;
  std::unique_ptr<char[]> arg_buffer_;
  std::unique_ptr<char[]> result_buffer_;
  const StructType *ret_type_;

 public:
  size_t arg_buffer_size{0};
  const StructType *args_type{nullptr};
  size_t result_buffer_size{0};

  // Note that I've tried to group `array_runtime_size` and
  // `is_device_allocations` into a small struct. However, it caused some test
  // cases to stuck.

  // `array_runtime_size` records the runtime size of the
  // corresponding array arguments.
  std::
      unordered_map<std::vector<int>, uint64, hashing::Hasher<std::vector<int>>>
          array_runtime_sizes;
  // `device_allocation_type` is set iff i-th arg is a `DeviceAllocation*`,
  // otherwise it is set to DevAllocType::kNone
  std::unordered_map<std::vector<int>,
                     DevAllocType,
                     hashing::Hasher<std::vector<int>>>
      device_allocation_type;

  std::unordered_map<std::pair<int, int>,
                     void *,
                     hashing::Hasher<std::pair<int, int>>>
      array_ptrs;
};

}  // namespace gstaichi::lang
