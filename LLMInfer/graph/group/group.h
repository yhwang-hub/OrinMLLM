#pragma once

#include <string>
#include <vector>

#include "graph/element/element_manager.h"

namespace orinmllm::graph {

class Group : public Element {
 public:
	Group() = default;
	~Group() override = default;

	template <typename T, typename = std::enable_if_t<std::is_base_of_v<Element, T>>>
	bool create_inner(const std::string& name, T** const out_element) {
		RETURN_VAL_IF(out_element == nullptr, false);
		return element_manager_.create<T>(name, out_element);
	}

	bool add_inner_depend(const std::string& element_name, const std::string& depend_name);
	bool init() override;
	bool process() override;
	bool deinit() override;
	ElementManager* element_manager();
	const ElementManager* element_manager() const;

 private:
	ElementManager element_manager_;
};

}  // namespace orinmllm::graph
