#include "graph/pipeline/pipeline_factory.h"

namespace orinmllm::graph {

std::unique_ptr<Pipeline> PipelineFactory::create(const std::string& name) {
	auto pipeline = std::make_unique<Pipeline>();
	pipeline->set_name(name);
	return pipeline;
}

}  // namespace orinmllm::graph
