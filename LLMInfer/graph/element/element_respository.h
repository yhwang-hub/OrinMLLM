#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "graph/element/element_relation.h"

namespace orinmllm::graph {

class ElementRepository {
 public:
	ElementRepository() = default;
	~ElementRepository() = default;

	bool add(std::unique_ptr<Element> element, Element** const out_element);
	bool remove(const std::string& name);
	Element* get(const std::string& name) const;
	std::vector<Element*> elements() const;
	std::size_t size() const;
	bool empty() const;
	void clear();

 private:
	std::vector<std::unique_ptr<Element>> elements_;
	std::unordered_map<std::string, Element*> name_map_;
};

}  // namespace orinmllm::graph
