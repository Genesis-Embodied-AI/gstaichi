#pragma once
#include <mutex>
#include <vector>
#include <memory>
#include <map>

#include "gstaichi/rhi/arch.h"
#include "gstaichi/rhi/device.h"

namespace gstaichi::lang {

class HostMemoryPool;

// This class can only be accessed by MemoryPool
class UnifiedAllocator {
 public:
  struct MemoryChunk {
    bool is_exclusive;
    void *data;
    void *head;
    void *tail;
  };

 private:
  static std::size_t default_allocator_size;

  UnifiedAllocator();

  void *allocate(std::size_t size,
                 std::size_t alignment,
                 bool exclusive = false);

  bool release(size_t sz, void *ptr);

  std::vector<MemoryChunk> chunks_;

  friend class HostMemoryPool;
  friend class HostMemoryPoolTestHelper;
};

}  // namespace gstaichi::lang
