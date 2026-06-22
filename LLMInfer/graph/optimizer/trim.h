#pragma once

#include <string>
#include <vector>

#include "graph/optimizer/optimizer.h"

namespace orinmllm::graph {

class TrimOptimizer : public Optimizer {
 public:
	TrimOptimizer() = default;
	~TrimOptimizer() override = default;

	bool optimize(ElementManager* const manager) override {
		RETURN_VAL_IF(manager == nullptr, false);
		std::vector<Element*> order;
		return manager->topo_sort(&order);
	}
};

}  // namespace orinmllm::graph
