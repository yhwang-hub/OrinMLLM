#pragma once

namespace orinmllm::graph {

enum class AspectType {
	kBeginInit = 0,
	kFinishInit = 1,
	kBeginRun = 2,
	kFinishRun = 3,
	kBeginDeinit = 4,
	kFinishDeinit = 5,
};

}  // namespace orinmllm::graph

