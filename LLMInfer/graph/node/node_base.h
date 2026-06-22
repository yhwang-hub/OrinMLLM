#pragma once

#include <functional>

#include "graph/element/element_relation.h"

namespace orinmllm::graph {

class NodeBase : public Element {
 public:
	NodeBase() = default;
	~NodeBase() override = default;
};

class LambdaNode : public NodeBase {
 public:
	using ProcessFunc = std::function<bool()>;

	explicit LambdaNode(ProcessFunc func = nullptr);
	~LambdaNode() override = default;

	bool process() override;
	bool set_process_func(const ProcessFunc& func);

 private:
	ProcessFunc process_func_;
};

}  // namespace orinmllm::graph
