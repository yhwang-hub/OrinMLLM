#include "graph/param/param.h"

#include <algorithm>

namespace orinmllm::graph {

std::vector<std::string> Param::backtrace() const {
  std::shared_lock<std::shared_mutex> lock(param_shared_lock_);
  return backtrace_;
}

bool Param::add_backtrace(const std::string& trace) {
  if (!enable_backtrace_) {
    return false;
  }
  std::unique_lock<std::shared_mutex> lock(param_shared_lock_);
  const auto iter = std::find(backtrace_.begin(), backtrace_.end(), trace);
  if (iter == backtrace_.end()) {
    backtrace_.push_back(trace);
  }
  return true;
}

void Param::clear_backtrace() {
  if (!enable_backtrace_) {
    return;
  }
  std::unique_lock<std::shared_mutex> lock(param_shared_lock_);
  backtrace_.clear();
}

const std::string& Param::key() const { return key_; }

void Param::set_key(const std::string& key) { key_ = key; }

bool Param::set_backtrace_enable(const bool enable_backtrace) {
	enable_backtrace_ = enable_backtrace;
	return true;
}

bool Param::backtrace_enable() const { return enable_backtrace_; }

}  // namespace orinmllm::graph

