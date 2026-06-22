#pragma once

#include <string>

#include "graph/aspect/aspect_define.h"

namespace orinmllm::graph {

inline std::string AspectTypeToString(const AspectType type) {
	switch (type) {
		case AspectType::kBeginInit:
			return "begin_init";
		case AspectType::kFinishInit:
			return "finish_init";
		case AspectType::kBeginRun:
			return "begin_run";
		case AspectType::kFinishRun:
			return "finish_run";
		case AspectType::kBeginDeinit:
			return "begin_deinit";
		case AspectType::kFinishDeinit:
			return "finish_deinit";
		default:
			return "unknown";
	}
}

}  // namespace orinmllm::graph
