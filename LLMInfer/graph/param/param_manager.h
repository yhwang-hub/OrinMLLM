#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "graph/param/param.h"
#include "graph/graph_common.h"

namespace orinmllm::graph {

class ParamManager : public ParamObj {
 public:
  ParamManager() = default;
  ~ParamManager() override;

	template <typename T, typename = std::enable_if_t<std::is_base_of_v<Param, T>>>
	bool create(const std::string& key, const bool backtrace = false) {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto iter = params_map_.find(key);
    if (iter != params_map_.end()) {
      return dynamic_cast<T*>(iter->second.get()) != nullptr;
    }

    auto param = std::make_unique<T>();
    param->set_key(key);
    param->set_backtrace_enable(backtrace);
    params_map_.emplace(key, std::move(param));
    return true;
  }

	template <typename T, typename = std::enable_if_t<std::is_base_of_v<Param, T>>>
	T* get(const std::string& key) const {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto iter = params_map_.find(key);
    if (iter == params_map_.end()) {
      return nullptr;
    }
    return dynamic_cast<T*>(iter->second.get());
  }

	bool remove_by_key(const std::string& key);
	std::vector<std::string> keys() const;

	bool init() override;
	bool deinit() override;
	bool clear();
	bool setup();
	bool reset();

 private:
  mutable std::mutex mutex_;
  std::unordered_map<std::string, std::unique_ptr<Param>> params_map_;
};

}  // namespace orinmllm::graph

