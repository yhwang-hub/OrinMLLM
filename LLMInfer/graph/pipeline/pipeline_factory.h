#pragma once

#include <memory>
#include <string>

#include "graph/pipeline/pipeline.h"

namespace orinmllm::graph {

class PipelineFactory {
 public:
	PipelineFactory() = default;
	~PipelineFactory() = default;

	static std::unique_ptr<Pipeline> create(const std::string& name);
};

}  // namespace orinmllm::graph
