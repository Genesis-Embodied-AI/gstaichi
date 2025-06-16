#include <string_view>
#include <filesystem>
#pragma once

constexpr std::string_view DUMP_IR_ENV = "TAICHI_DUMP_IR";
constexpr std::string_view LOAD_IR_ENV = "TAICHI_LOAD_IR";
const std::filesystem::path IR_DUMP_DIR = std::filesystem::path("/tmp/ir/");
