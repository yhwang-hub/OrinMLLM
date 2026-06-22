#include "graph/element/element_respository.h"

#include <algorithm>

namespace orinmllm::graph {

bool ElementRepository::add(std::unique_ptr<Element> element, Element** const out_element) {
	RETURN_VAL_IF(element == nullptr || out_element == nullptr, false);
	const std::string element_name = element->name();
	RETURN_VAL_IF(element_name.empty(), false);
	RETURN_VAL_IF(name_map_.find(element_name) != name_map_.end(), false);
	Element* const raw_element = element.get();
	elements_.push_back(std::move(element));
	name_map_.emplace(element_name, raw_element);
	*out_element = raw_element;
	return true;
}

bool ElementRepository::remove(const std::string& name) {
	const auto map_iter = name_map_.find(name);
	RETURN_VAL_IF(map_iter == name_map_.end(), false);
	Element* const removed = map_iter->second;
	for (const auto& element : elements_) {
		if (element != nullptr && element.get() != removed) {
			element->remove_depend(removed);
		}
	}
	name_map_.erase(map_iter);
	const auto elem_iter = std::remove_if(elements_.begin(), elements_.end(),
																			 [removed](const std::unique_ptr<Element>& element) {
																				 return element.get() == removed;
																			 });
	elements_.erase(elem_iter, elements_.end());
	return true;
}

Element* ElementRepository::get(const std::string& name) const {
	const auto iter = name_map_.find(name);
	return iter == name_map_.end() ? nullptr : iter->second;
}

std::vector<Element*> ElementRepository::elements() const {
	std::vector<Element*> result;
	result.reserve(elements_.size());
	for (const auto& element : elements_) {
		if (element != nullptr) {
			result.push_back(element.get());
		}
	}
	return result;
}

std::size_t ElementRepository::size() const { return elements_.size(); }

bool ElementRepository::empty() const { return elements_.empty(); }

void ElementRepository::clear() {
	name_map_.clear();
	elements_.clear();
}

}  // namespace orinmllm::graph
