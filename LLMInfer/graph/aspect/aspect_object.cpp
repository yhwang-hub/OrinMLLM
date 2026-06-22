#include "graph/aspect/aspect_object.h"

#include "graph/element/element_relation.h"

namespace orinmllm::graph {

AspectObject* AspectObject::set_belong(Element* const belong) {
  RETURN_VAL_IF(belong == nullptr, this);
  belong_ = belong;
  return this;
}

Element* AspectObject::belong() const { return belong_; }

const std::string& AspectObject::get_name() const {
  if (!name_.empty()) {
    return name_;
  }
  static const std::string kUnknownName = "unknown";
  return kUnknownName;
}

}  // namespace orinmllm::graph

