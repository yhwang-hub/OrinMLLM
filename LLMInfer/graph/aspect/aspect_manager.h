#pragma once

#include <memory>
#include <type_traits>
#include <vector>

#include "graph/aspect/aspect.h"
#include "graph/aspect/aspect_define.h"
#include "graph/graph_common.h"

namespace orinmllm::graph {

class Element;

class AspectManager {
 public:
	AspectManager() = default;
	~AspectManager() = default;

	AspectManager(const AspectManager&) = delete;
	AspectManager& operator=(const AspectManager&) = delete;

	template <typename T, typename = std::enable_if_t<std::is_base_of_v<Aspect, T>>>
	bool add(Element* const belong) {
		RETURN_VAL_IF(belong == nullptr, false);
		auto aspect = std::make_unique<T>();
		aspect->set_belong(belong);
		aspects_.push_back(std::move(aspect));
		return true;
	}

	bool add_aspect(std::unique_ptr<Aspect> aspect, Element* const belong);
	bool trigger(const AspectType type, const bool is_run_success = true);
	std::size_t size() const;
	bool empty() const;
	void clear();

 private:
	std::vector<std::unique_ptr<Aspect>> aspects_;
};

}  // namespace orinmllm::graph
