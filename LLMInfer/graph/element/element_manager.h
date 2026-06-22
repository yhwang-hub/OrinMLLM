#pragma once

#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include "graph/element/element_respository.h"

namespace orinmllm::graph {

class ElementManager {
 public:
	ElementManager();
	~ElementManager() = default;

	template <typename T, typename = std::enable_if_t<std::is_base_of_v<Element, T>>>
	bool create(const std::string& name, T** const out_element) {
		RETURN_VAL_IF(out_element == nullptr, false);
		auto element = std::make_unique<T>();
		element->set_name(name);
		element->set_param_manager(param_manager_);
		Element* raw_element = nullptr;
		RETURN_VAL_IF(!repository_.add(std::move(element), &raw_element), false);
		*out_element = dynamic_cast<T*>(raw_element);
		return *out_element != nullptr;
	}

	bool add_depend(const std::string& element_name, const std::string& depend_name);
	Element* get(const std::string& name) const;
	std::vector<Element*> elements() const;
	bool topo_sort(std::vector<Element*>* const out_order) const;
	bool has_cycle() const;
	std::size_t size() const;
	void clear();

	std::shared_ptr<ParamManager> param_manager() const;
	bool set_param_manager(const std::shared_ptr<ParamManager>& param_manager);

 private:
	ElementRepository repository_;
	std::shared_ptr<ParamManager> param_manager_;
};

}  // namespace orinmllm::graph
