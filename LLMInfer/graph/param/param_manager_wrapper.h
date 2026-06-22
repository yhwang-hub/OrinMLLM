#pragma once

#include <memory>
#include <set>
#include <string>
#include <type_traits>
#include <vector>

#include "graph/param/param_manager.h"
#include "graph/graph_common.h"

namespace orinmllm::graph {

class ParamManagerWrapper {
 public:
	virtual ~ParamManagerWrapper() = default;

	template <typename T, typename = std::enable_if_t<std::is_base_of_v<Param, T>>>
	bool create_param(const std::string& key, const bool backtrace = false) {
		RETURN_VAL_IF(param_manager_ == nullptr, false);
		return param_manager_->create<T>(key, backtrace);
	}

	template <typename T, typename = std::enable_if_t<std::is_base_of_v<Param, T>>>
	T* get_param(const std::string& key) {
		RETURN_VAL_IF(param_manager_ == nullptr, nullptr);
		T* const param = param_manager_->get<T>(key);
		if (param != nullptr) {
			concerned_params_.insert(param);
		}
		return param;
	}

	template <typename T, typename = std::enable_if_t<std::is_base_of_v<Param, T>>>
	T* get_param_or_null(const std::string& key) {
		return get_param<T>(key);
	}

	bool remove_param(const std::string& key) {
		RETURN_VAL_IF(param_manager_ == nullptr, false);
		return param_manager_->remove_by_key(key);
	}

	std::vector<std::string> param_keys() const {
		if (param_manager_ == nullptr) {
			return {};
		}
		return param_manager_->keys();
	}

	std::vector<std::string> concerned_param_keys() const {
		std::vector<std::string> keys;
		keys.reserve(concerned_params_.size());
		for (const Param* const param : concerned_params_) {
			if (param != nullptr) {
				keys.push_back(param->key());
			}
		}
		return keys;
	}

	void set_param_manager(const std::shared_ptr<ParamManager>& param_manager) {
		param_manager_ = param_manager;
	}

 protected:
	std::shared_ptr<ParamManager> param_manager_;

 private:
	std::set<Param*> concerned_params_;
};

}  // namespace orinmllm::graph

