#include "graph/element/element_manager.h"

#include <queue>
#include <unordered_map>

namespace orinmllm::graph {

ElementManager::ElementManager() : param_manager_(std::make_shared<ParamManager>()) {}

bool ElementManager::add_depend(const std::string& element_name, const std::string& depend_name) {
	Element* const element = repository_.get(element_name);
	Element* const depend = repository_.get(depend_name);
	RETURN_VAL_IF(element == nullptr || depend == nullptr, false);
	return element->add_depend(depend);
}

Element* ElementManager::get(const std::string& name) const { return repository_.get(name); }

std::vector<Element*> ElementManager::elements() const { return repository_.elements(); }

bool ElementManager::topo_sort(std::vector<Element*>* const out_order) const {
	RETURN_VAL_IF(out_order == nullptr, false);
	out_order->clear();
	const std::vector<Element*> all_elements = repository_.elements();
	std::unordered_map<Element*, std::size_t> indegree;
	for (Element* const element : all_elements) {
		RETURN_VAL_IF(element == nullptr, false);
		indegree[element] = element->depends().size();
	}

	std::queue<Element*> ready;
	for (const auto& item : indegree) {
		if (item.second == 0) {
			ready.push(item.first);
		}
	}

	while (!ready.empty()) {
		Element* const element = ready.front();
		ready.pop();
		out_order->push_back(element);
		for (Element* const successor : element->successors()) {
			RETURN_VAL_IF(successor == nullptr, false);
			auto iter = indegree.find(successor);
			RETURN_VAL_IF(iter == indegree.end() || iter->second == 0, false);
			--iter->second;
			if (iter->second == 0) {
				ready.push(successor);
			}
		}
	}

	return out_order->size() == all_elements.size();
}

bool ElementManager::has_cycle() const {
	std::vector<Element*> order;
	return !topo_sort(&order);
}

std::size_t ElementManager::size() const { return repository_.size(); }

void ElementManager::clear() { repository_.clear(); }

std::shared_ptr<ParamManager> ElementManager::param_manager() const { return param_manager_; }

bool ElementManager::set_param_manager(const std::shared_ptr<ParamManager>& param_manager) {
	RETURN_VAL_IF(param_manager == nullptr, false);
	param_manager_ = param_manager;
	for (Element* const element : repository_.elements()) {
		RETURN_VAL_IF(element == nullptr, false);
		element->set_param_manager(param_manager_);
	}
	return true;
}

}  // namespace orinmllm::graph
