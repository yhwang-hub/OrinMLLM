#include "graph/param/param_manager.h"

namespace orinmllm::graph {

ParamManager::~ParamManager() { clear(); }

bool ParamManager::init() {
	std::lock_guard<std::mutex> lock(mutex_);
	for (const auto& item : params_map_) {
		RETURN_VAL_IF(item.second == nullptr, false);
		RETURN_VAL_IF(!item.second->init(), false);
	}
	return true;
}

bool ParamManager::deinit() {
	std::lock_guard<std::mutex> lock(mutex_);
	for (const auto& item : params_map_) {
		RETURN_VAL_IF(item.second == nullptr, false);
		RETURN_VAL_IF(!item.second->deinit(), false);
	}
	params_map_.clear();
	return true;
}

bool ParamManager::clear() {
	std::lock_guard<std::mutex> lock(mutex_);
	params_map_.clear();
	return true;
}

bool ParamManager::setup() {
	std::lock_guard<std::mutex> lock(mutex_);
	for (const auto& item : params_map_) {
		RETURN_VAL_IF(item.second == nullptr, false);
		RETURN_VAL_IF(!item.second->setup(), false);
	}
	return true;
}

bool ParamManager::reset() {
	std::lock_guard<std::mutex> lock(mutex_);
	for (const auto& item : params_map_) {
		RETURN_VAL_IF(item.second == nullptr, false);
		RETURN_VAL_IF(!item.second->reset(), false);
	}
	return true;
}

bool ParamManager::remove_by_key(const std::string& key) {
	std::lock_guard<std::mutex> lock(mutex_);
	const auto iter = params_map_.find(key);
	if (iter == params_map_.end()) {
		return false;
	}
	params_map_.erase(iter);
	return true;
}

std::vector<std::string> ParamManager::keys() const {
	std::lock_guard<std::mutex> lock(mutex_);
	std::vector<std::string> result;
	result.reserve(params_map_.size());
	for (const auto& item : params_map_) {
		result.push_back(item.first);
	}
	return result;
}

}  // namespace orinmllm::graph

