#include "graph/node/node_base.h"

namespace orinmllm::graph {

LambdaNode::LambdaNode(ProcessFunc func) : process_func_(std::move(func)) {}

bool LambdaNode::process() {
	if (process_func_ == nullptr) {
		return true;
	}
	return process_func_();
}

bool LambdaNode::set_process_func(const ProcessFunc& func) {
	process_func_ = func;
	return true;
}

}  // namespace orinmllm::graph
