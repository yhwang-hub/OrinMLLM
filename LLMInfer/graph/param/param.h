#pragma once

#include <mutex>
#include <shared_mutex>
#include <string>
#include <vector>

#include "graph/gobj.h"

namespace orinmllm::graph {

class Param : public ParamObj {
 public:
  Param() = default;
  ~Param() override = default;

  std::shared_mutex& param_shared_lock() { return param_shared_lock_; }

  std::vector<std::string> backtrace() const;
  bool add_backtrace(const std::string& trace);
  void clear_backtrace();

  const std::string& key() const;
  void set_key(const std::string& key);

  bool set_backtrace_enable(const bool enable_backtrace);
  bool backtrace_enable() const;

  virtual bool setup() { return true; }
  virtual bool reset() { return true; }

 private:
  mutable std::shared_mutex param_shared_lock_;
  bool enable_backtrace_ = false;
  std::string key_;
  std::vector<std::string> backtrace_;
};

}  // namespace orinmllm::graph

