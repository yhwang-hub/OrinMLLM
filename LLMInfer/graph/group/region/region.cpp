#include "graph/group/region/region.h"

#include <algorithm>

namespace orinmllm::graph {

bool Region::add_entry(Element* const element) {
	RETURN_VAL_IF(element == nullptr, false);
	if (std::find(entries_.begin(), entries_.end(), element) == entries_.end()) {
		entries_.push_back(element);
	}
	return true;
}

bool Region::add_exit(Element* const element) {
	RETURN_VAL_IF(element == nullptr, false);
	if (std::find(exits_.begin(), exits_.end(), element) == exits_.end()) {
		exits_.push_back(element);
	}
	return true;
}

const std::vector<Element*>& Region::entries() const { return entries_; }

const std::vector<Element*>& Region::exits() const { return exits_; }

}  // namespace orinmllm::graph
