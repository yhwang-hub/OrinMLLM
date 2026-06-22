#pragma once

#include "graph/element/element_manager.h"

namespace orinmllm::graph {

class Optimizer {
 public:
	Optimizer() = default;
	virtual ~Optimizer() = default;

	virtual bool optimize(ElementManager* const manager) = 0;
};

}  // namespace orinmllm::graph
