#pragma once

#include <string>
#include <vector>

#include "graph/element/element_manager.h"

namespace orinmllm::graph {

class Engine {
 public:
	Engine() = default;
	~Engine() = default;

	bool init(const std::vector<Element*>& elements);
	bool run(const std::vector<Element*>& elements);
	bool deinit(const std::vector<Element*>& elements);
};

}  // namespace orinmllm::graph
