#pragma once

#include <string>

#include "gstaichi/rhi/arch.h"

namespace gstaichi::lang {

struct CompileConfig;
struct DeviceCapabilityConfig;
class Program;
class IRNode;
class SNode;
class Kernel;

std::string get_hashed_offline_cache_key_of_snode(const SNode *snode);
std::string get_hashed_offline_cache_key(const CompileConfig &config,
                                         const DeviceCapabilityConfig &caps,
                                         Kernel *kernel);
void gen_offline_cache_key(IRNode *ast, std::ostream *os);

}  // namespace gstaichi::lang
