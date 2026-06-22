#pragma once

#include <memory>
#include <set>
#include <string>
#include <vector>

#include "graph/aspect/aspect_manager.h"
#include "graph/descinfo.h"
#include "graph/gobj.h"
#include "graph/graph_common.h"

namespace orinmllm::graph {

class Element : public Runnable, public DescInfo, public ParamManagerWrapper {
 public:
	Element();
	~Element() override = default;

	Element(const Element&) = delete;
	Element& operator=(const Element&) = delete;

	bool add_depend(Element* const element);
	bool remove_depend(Element* const element);
	bool has_depend(const Element* const element) const;
	const std::vector<Element*>& depends() const;
	const std::set<Element*>& successors() const;

	bool init() override;
	bool run() override;
	bool deinit() override;

	virtual bool process() { return true; }

	AspectManager* aspect_manager();
	const AspectManager* aspect_manager() const;
	ElementState state() const;
	bool is_runnable() const;
	std::string DebugString() const;

 private:
	friend class ElementManager;

	bool add_successor(Element* const element);
	bool remove_successor(Element* const element);

	std::vector<Element*> depends_;
	std::set<Element*> successors_;
	AspectManager aspect_manager_;
	ElementState state_ = ElementState::kCreated;
};

using ElementPtr = std::unique_ptr<Element>;

}  // namespace orinmllm::graph
