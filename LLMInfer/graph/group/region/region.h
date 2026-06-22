#pragma once

#include <vector>

#include "graph/group/group.h"

namespace orinmllm::graph {

class Region : public Group {
 public:
	Region() = default;
	~Region() override = default;

	bool add_entry(Element* const element);
	bool add_exit(Element* const element);
	const std::vector<Element*>& entries() const;
	const std::vector<Element*>& exits() const;

 private:
	std::vector<Element*> entries_;
	std::vector<Element*> exits_;
};

}  // namespace orinmllm::graph
