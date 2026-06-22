#include "graph/pipeline/pipeline.h"

#include "graph/optimizer/optimizer.h"

namespace orinmllm::graph {

Pipeline::Pipeline() {
	set_name(GenerateSessionId("pipeline"));
	set_param_manager(element_manager_.param_manager());
}

Pipeline::~Pipeline() {
	if (is_initialized_) {
		(void)deinit();
	}
}

bool Pipeline::add_depend(const std::string& element_name, const std::string& depend_name) {
	return element_manager_.add_depend(element_name, depend_name);
}

bool Pipeline::init() {
	RETURN_VAL_IF(!element_manager_.topo_sort(&run_order_), false);
	RETURN_VAL_IF(!element_manager_.param_manager()->init(), false);
	RETURN_VAL_IF(!element_manager_.param_manager()->setup(), false);
	RETURN_VAL_IF(!engine_.init(run_order_), false);
	state_ = PipelineState::kInitialized;
	is_initialized_ = true;
	return true;
}

bool Pipeline::run() {
	if (!is_initialized_) {
		RETURN_VAL_IF(!init(), false);
	}
	state_ = PipelineState::kRunning;
	RETURN_VAL_IF(!engine_.run(run_order_), false);
	state_ = PipelineState::kFinished;
	return true;
}

bool Pipeline::deinit() {
	if (!is_initialized_) {
		return true;
	}
	RETURN_VAL_IF(!engine_.deinit(run_order_), false);
	RETURN_VAL_IF(!element_manager_.param_manager()->reset(), false);
	RETURN_VAL_IF(!element_manager_.param_manager()->deinit(), false);
	state_ = PipelineState::kDeinitialized;
	is_initialized_ = false;
	return true;
}

bool Pipeline::dump(std::string* const out_text) const {
	RETURN_VAL_IF(out_text == nullptr, false);
	std::vector<Element*> order;
	RETURN_VAL_IF(!element_manager_.topo_sort(&order), false);
	std::ostringstream oss;
	oss << "Pipeline(" << name() << ")\n";
	for (const Element* const element : order) {
		RETURN_VAL_IF(element == nullptr, false);
		oss << "  " << element->DebugString() << "\n";
	}
	*out_text = oss.str();
	return true;
}

bool Pipeline::optimize(Optimizer* const optimizer) {
	RETURN_VAL_IF(optimizer == nullptr, false);
	return optimizer->optimize(&element_manager_);
}

Element* Pipeline::get(const std::string& name) const { return element_manager_.get(name); }

PipelineState Pipeline::state() const { return state_; }

std::size_t Pipeline::size() const { return element_manager_.size(); }

ElementManager* Pipeline::element_manager() { return &element_manager_; }

const ElementManager* Pipeline::element_manager() const { return &element_manager_; }

}  // namespace orinmllm::graph
