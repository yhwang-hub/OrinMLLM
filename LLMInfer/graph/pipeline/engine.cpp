#include "graph/pipeline/engine.h"

namespace orinmllm::graph {

bool Engine::init(const std::vector<Element*>& elements) {
	for (Element* const element : elements) {
		RETURN_VAL_IF(element == nullptr, false);
		RETURN_VAL_IF(!element->init(), false);
	}
	return true;
}

bool Engine::run(const std::vector<Element*>& elements) {
	for (Element* const element : elements) {
		RETURN_VAL_IF(element == nullptr, false);
		RETURN_VAL_IF(!element->run(), false);
	}
	return true;
}

bool Engine::deinit(const std::vector<Element*>& elements) {
	for (auto iter = elements.rbegin(); iter != elements.rend(); ++iter) {
		RETURN_VAL_IF(*iter == nullptr, false);
		RETURN_VAL_IF(!(*iter)->deinit(), false);
	}
	return true;
}

}  // namespace orinmllm::graph
