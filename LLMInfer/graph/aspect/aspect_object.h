#pragma once

#include <string>

#include "graph/descinfo.h"
#include "graph/gobj.h"
#include "graph/param/param_manager_wrapper.h"

namespace orinmllm::graph {

class Element;

class AspectObject : public GObj, public DescInfo, public ParamManagerWrapper {
 public:
	AspectObject() = default;
	~AspectObject() override = default;

	AspectObject* set_belong(Element* const belong);
	Element* belong() const;
	const std::string& get_name() const;

	bool run() override { return false; }

 protected:
	Element* belong_ = nullptr;
};

}  // namespace orinmllm::graph

