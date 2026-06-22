#pragma once

#include <queue>
#include <set>
#include <vector>

#include "graph/optimizer/optimizer.h"

namespace orinmllm::graph {

class SeparateOptimizer : public Optimizer {
 public:
	SeparateOptimizer() = default;
	~SeparateOptimizer() override = default;

	bool optimize(ElementManager* const manager) override {
		RETURN_VAL_IF(manager == nullptr, false);
		components_.clear();
		std::set<Element*> visited;
		for (Element* const root : manager->elements()) {
			RETURN_VAL_IF(root == nullptr, false);
			if (visited.find(root) != visited.end()) {
				continue;
			}
			std::vector<Element*> component;
			std::queue<Element*> pending;
			pending.push(root);
			visited.insert(root);
			while (!pending.empty()) {
				Element* const current = pending.front();
				pending.pop();
				component.push_back(current);
				for (Element* const depend : current->depends()) {
					PushIfNew(depend, &visited, &pending);
				}
				for (Element* const successor : current->successors()) {
					PushIfNew(successor, &visited, &pending);
				}
			}
			components_.push_back(component);
		}
		return true;
	}

	const std::vector<std::vector<Element*>>& components() const { return components_; }

 private:
	static bool PushIfNew(Element* const element, std::set<Element*>* const visited,
												std::queue<Element*>* const pending) {
		RETURN_VAL_IF(element == nullptr || visited == nullptr || pending == nullptr, false);
		if (visited->find(element) == visited->end()) {
			visited->insert(element);
			pending->push(element);
		}
		return true;
	}

	std::vector<std::vector<Element*>> components_;
};

}  // namespace orinmllm::graph
