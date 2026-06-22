#include "graph/group/group.h"

namespace orinmllm::graph {

bool Group::add_inner_depend(const std::string& element_name, const std::string& depend_name) {
	return element_manager_.add_depend(element_name, depend_name);
}

bool Group::init() {
	RETURN_VAL_IF(!Element::init(), false);
	std::vector<Element*> order;
	RETURN_VAL_IF(!element_manager_.topo_sort(&order), false);
	for (Element* const element : order) {
		RETURN_VAL_IF(element == nullptr, false);
		RETURN_VAL_IF(!element->init(), false);
	}
	return true;
}

bool Group::process() {
	std::vector<Element*> order;
	RETURN_VAL_IF(!element_manager_.topo_sort(&order), false);
	for (Element* const element : order) {
		RETURN_VAL_IF(element == nullptr, false);
		RETURN_VAL_IF(!element->run(), false);
	}
	return true;
}

bool Group::deinit() {
	std::vector<Element*> order;
	RETURN_VAL_IF(!element_manager_.topo_sort(&order), false);
	for (auto iter = order.rbegin(); iter != order.rend(); ++iter) {
		RETURN_VAL_IF(*iter == nullptr, false);
		RETURN_VAL_IF(!(*iter)->deinit(), false);
	}
	return Element::deinit();
}

ElementManager* Group::element_manager() { return &element_manager_; }

const ElementManager* Group::element_manager() const { return &element_manager_; }

}  // namespace orinmllm::graph
