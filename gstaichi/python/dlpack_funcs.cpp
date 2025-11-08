#include "dlpack_funcs.h"

#include "gstaichi/program/ndarray.h"
#include "gstaichi/program/program.h"
#include "gstaichi/rhi/cuda/cuda_device.h"

namespace gstaichi {
namespace lang {
pybind11::capsule dlpack_dump_ndarray_info(Program *program, Ndarray *ndarray) {
    std::cout << "dump_ndarray_info" << std::endl;
  std::cout << ndarray->read_int({0, 0}) << std::endl;
  // int *data_ptr = reinterpret_cast<int *>(get_ndarray_data_ptr
  
  DeviceAllocation devalloc = ndarray->get_device_allocation();
  // using cuda = gstaichi::lang::cuda;
  cuda::CudaDevice *cuda_device = dynamic_cast<cuda::CudaDevice *>(devalloc.device);
  std::cout << " not nullptr " << (cuda_device != nullptr) << std::endl;

    MyData *my_data = new MyData;
    my_data->value = 31;

    // return my_data;

    auto deleter = [](PyObject *capsule) {
        MyData *my_data = static_cast<MyData *>(PyCapsule_GetPointer(capsule, "my_data"));
        delete my_data;
    };

    pybind11::capsule capsule = pybind11::capsule(static_cast<void *>(my_data), "my_data", deleter);
    return capsule;
}
}
}
