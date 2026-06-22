#include "graph/aspect/aspect_manager.h"

namespace orinmllm::graph {

bool AspectManager::add_aspect(std::unique_ptr<Aspect> aspect, Element* const belong) {
	RETURN_VAL_IF(aspect == nullptr || belong == nullptr, false);
	aspect->set_belong(belong);
	aspects_.push_back(std::move(aspect));
	return true;
}

bool AspectManager::trigger(const AspectType type, const bool is_run_success) {
	for (const auto& aspect : aspects_) {
		RETURN_VAL_IF(aspect == nullptr, false);
		switch (type) {
			case AspectType::kBeginInit:
				RETURN_VAL_IF(!aspect->begin_init(), false);
				break;
			case AspectType::kFinishInit:
				RETURN_VAL_IF(!aspect->finish_init(), false);
				break;
			case AspectType::kBeginRun:
				RETURN_VAL_IF(!aspect->begin_run(), false);
				break;
			case AspectType::kFinishRun:
				RETURN_VAL_IF(!aspect->finish_run(is_run_success), false);
				break;
			case AspectType::kBeginDeinit:
				RETURN_VAL_IF(!aspect->begin_deinit(), false);
				break;
			case AspectType::kFinishDeinit:
				RETURN_VAL_IF(!aspect->finish_deinit(), false);
				break;
			default:
				return false;
		}
	}
	return true;
}

std::size_t AspectManager::size() const { return aspects_.size(); }

bool AspectManager::empty() const { return aspects_.empty(); }

void AspectManager::clear() { aspects_.clear(); }

}  // namespace orinmllm::graph
