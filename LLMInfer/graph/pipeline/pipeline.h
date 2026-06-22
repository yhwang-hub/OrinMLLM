#pragma once

#include <memory>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#include "graph/element/element_manager.h"
#include "graph/pipeline/engine.h"

namespace orinmllm::graph {

class Optimizer;

class Pipeline : public DescInfo, public ParamManagerWrapper {
 public:
	Pipeline();
	~Pipeline();

	Pipeline(const Pipeline&) = delete;
	Pipeline& operator=(const Pipeline&) = delete;

	template <typename T, typename = std::enable_if_t<std::is_base_of_v<Element, T>>>
	bool create(const std::string& name, T** const out_element) {
		RETURN_VAL_IF(out_element == nullptr, false);
		return element_manager_.create<T>(name, out_element);
	}

	bool add_depend(const std::string& element_name, const std::string& depend_name);
	bool init();
	bool run();
	bool deinit();
	bool dump(std::string* const out_text) const;
	bool optimize(Optimizer* const optimizer);

	Element* get(const std::string& name) const;
	PipelineState state() const;
	std::size_t size() const;
	ElementManager* element_manager();
	const ElementManager* element_manager() const;

 private:
	ElementManager element_manager_;
	Engine engine_;
	std::vector<Element*> run_order_;
	PipelineState state_ = PipelineState::kCreated;
	bool is_initialized_ = false;
};

}  // namespace orinmllm::graph
