#include "graph/element/element_relation.h"

#include <algorithm>
#include <sstream>

namespace orinmllm::graph {

Element::Element() { set_session(GenerateSessionId("element")); }

bool Element::add_depend(Element* const element) {
	RETURN_VAL_IF(element == nullptr || element == this, false);
	if (has_depend(element)) {
		return true;
	}
	depends_.push_back(element);
	return element->add_successor(this);
}

bool Element::remove_depend(Element* const element) {
	RETURN_VAL_IF(element == nullptr, false);
	const auto iter = std::find(depends_.begin(), depends_.end(), element);
	RETURN_VAL_IF(iter == depends_.end(), false);
	depends_.erase(iter);
	return element->remove_successor(this);
}

bool Element::has_depend(const Element* const element) const {
	RETURN_VAL_IF(element == nullptr, false);
	return std::find(depends_.begin(), depends_.end(), element) != depends_.end();
}

const std::vector<Element*>& Element::depends() const { return depends_; }

const std::set<Element*>& Element::successors() const { return successors_; }

bool Element::init() {
	RETURN_VAL_IF(!aspect_manager_.trigger(AspectType::kBeginInit), false);
	state_ = ElementState::kInitialized;
	RETURN_VAL_IF(!aspect_manager_.trigger(AspectType::kFinishInit), false);
	return true;
}

bool Element::run() {
	RETURN_VAL_IF(!aspect_manager_.trigger(AspectType::kBeginRun), false);
	state_ = ElementState::kRunning;
	const bool is_process_success = process();
	state_ = is_process_success ? ElementState::kFinished : state_;
	RETURN_VAL_IF(!aspect_manager_.trigger(AspectType::kFinishRun, is_process_success), false);
	return is_process_success;
}

bool Element::deinit() {
	RETURN_VAL_IF(!aspect_manager_.trigger(AspectType::kBeginDeinit), false);
	state_ = ElementState::kDeinitialized;
	RETURN_VAL_IF(!aspect_manager_.trigger(AspectType::kFinishDeinit), false);
	return true;
}

AspectManager* Element::aspect_manager() { return &aspect_manager_; }

const AspectManager* Element::aspect_manager() const { return &aspect_manager_; }

ElementState Element::state() const { return state_; }

bool Element::is_runnable() const {
	return state_ == ElementState::kInitialized || state_ == ElementState::kFinished;
}

std::string Element::DebugString() const {
	std::ostringstream oss;
	oss << name() << " <- [";
	for (std::size_t index = 0; index < depends_.size(); ++index) {
		if (index > 0) {
			oss << ", ";
		}
		oss << (depends_[index] == nullptr ? "null" : depends_[index]->name());
	}
	oss << "]";
	return oss.str();
}

bool Element::add_successor(Element* const element) {
	RETURN_VAL_IF(element == nullptr || element == this, false);
	successors_.insert(element);
	return true;
}

bool Element::remove_successor(Element* const element) {
	RETURN_VAL_IF(element == nullptr, false);
	return successors_.erase(element) > 0;
}

}  // namespace orinmllm::graph
