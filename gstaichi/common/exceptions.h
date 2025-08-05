#pragma once

#include <exception>
#include <string>
#include <string_view>
#include "gstaichi/common/logging.h"

namespace gstaichi::lang {

class IRModified {};

struct Location {
  int line_number = 0;
  std::string var_name = "";
};

struct DebugInfo {
  Location src_loc;
  std::string tb;

  explicit DebugInfo() = default;

  explicit DebugInfo(std::string tb_) : tb(tb_) {
  }

  explicit DebugInfo(const char *tb_) : tb(tb_) {
  }

  std::string get_last_tb() const {
    return tb;
  }

  std::string const &get_tb() const {
    return tb;
  }

  void set_tb(std::string const &tb) {
    this->tb = tb;
  }
};

class GsTaichiExceptionImpl : public std::exception {
  friend struct ErrorEmitter;

 protected:
  std::string msg_;

 public:
  // Add default constructor to allow passing Exception to ErrorEmitter
  // TODO: remove this and find a better way
  explicit GsTaichiExceptionImpl() = default;
  explicit GsTaichiExceptionImpl(const std::string msg) : msg_(msg) {
  }
  const char *what() const noexcept override {
    return msg_.c_str();
  }
};

class GsTaichiError : public GsTaichiExceptionImpl {
  using GsTaichiExceptionImpl::GsTaichiExceptionImpl;
};

class GsTaichiWarning : public GsTaichiExceptionImpl {
  using GsTaichiExceptionImpl::GsTaichiExceptionImpl;

 protected:
  static constexpr std::string_view name_ = "GsTaichiWarning";

 public:
  void emit() {
    gstaichi::Logger::get_instance().warn(std::string(name_) + "\n" + msg_);
  }
};

class GsTaichiTypeError : public GsTaichiError {
  using GsTaichiError::GsTaichiError;
};

class GsTaichiSyntaxError : public GsTaichiError {
  using GsTaichiError::GsTaichiError;
};

class GsTaichiIndexError : public GsTaichiError {
  using GsTaichiError::GsTaichiError;
};

class GsTaichiRuntimeError : public GsTaichiError {
  using GsTaichiError::GsTaichiError;
};

class GsTaichiAssertionError : public GsTaichiError {
  using GsTaichiError::GsTaichiError;
};

class GsTaichiIrError : public GsTaichiError {
  using GsTaichiError::GsTaichiError;
};

class GsTaichiCastWarning : public GsTaichiWarning {
  using GsTaichiWarning::GsTaichiWarning;
  static constexpr std::string_view name_ = "GsTaichiCastWarning";
};

class GsTaichiTypeWarning : public GsTaichiWarning {
  using GsTaichiWarning::GsTaichiWarning;
  static constexpr std::string_view name_ = "GsTaichiTypeWarning";
};

class GsTaichiIrWarning : public GsTaichiWarning {
  using GsTaichiWarning::GsTaichiWarning;
  static constexpr std::string_view name_ = "GsTaichiIrWarning";
};

class GsTaichiIndexWarning : public GsTaichiWarning {
  using GsTaichiWarning::GsTaichiWarning;
  static constexpr std::string_view name_ = "GsTaichiIndexWarning";
};

class GsTaichiRuntimeWarning : public GsTaichiWarning {
  using GsTaichiWarning::GsTaichiWarning;
  static constexpr std::string_view name_ = "GsTaichiRuntimeWarning";
};

struct ErrorEmitter {
  ErrorEmitter() = delete;
  ErrorEmitter(ErrorEmitter &) = delete;
  ErrorEmitter(ErrorEmitter &&) = delete;

  // Emit an error on stmt with error message
  template <typename E,
            typename = std::enable_if_t<
                std::is_base_of_v<GsTaichiExceptionImpl, std::decay_t<E>>>,
            // The expected type for T is `Stmt`, `Expression`, or `DebugInfo`.
            // These types have a member function named get_tb() that returns
            // trace back information as a `std::string`.
            typename T,
            typename = std::enable_if_t<std::is_same_v<
                std::decay_t<decltype(std::declval<T>()->get_tb())>,
                std::string>>>
  ErrorEmitter(E &&error, T p_dbg_info, std::string &&error_msg) {
    if constexpr ((std::is_same_v<std::decay_t<T>, DebugInfo *> ||
                   std::is_same_v<std::decay_t<T>, const DebugInfo *>) &&
                  std::is_base_of_v<GsTaichiError, std::decay_t<E>>) {
      // Indicates a failed C++ API call from Python side, we should not print
      // tb here
      error.msg_ = error_msg;
    } else {
      error.msg_ = p_dbg_info->get_last_tb() + error_msg;
    }

    if constexpr (std::is_base_of_v<GsTaichiWarning, std::decay_t<E>>) {
      error.emit();
    } else if constexpr (std::is_base_of_v<GsTaichiError, std::decay_t<E>>) {
      throw std::move(error);
    } else {
      TI_NOT_IMPLEMENTED;
    }
  }
};

}  // namespace gstaichi::lang
