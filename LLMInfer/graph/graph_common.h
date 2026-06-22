#pragma once

#include <atomic>
#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

namespace orinmllm::graph {

#define RETURN_VAL_IF(condition, value) \
  do {                                   \
    if (condition) {                     \
      return value;                      \
    }                                    \
  } while (0)

#define RETURN_IF(condition) \
  do {                        \
    if (condition) {          \
      return;                 \
    }                         \
  } while (0)

inline std::string GenerateSessionId(const std::string& prefix) {
  static std::atomic<uint64_t> g_seq{0};
  std::ostringstream oss;
  oss << prefix << "-" << g_seq.fetch_add(1, std::memory_order_relaxed);
  return oss.str();
}

enum class ElementState {
  kCreated = 0,
  kInitialized = 1,
  kRunning = 2,
  kFinished = 3,
  kDeinitialized = 4,
};

enum class PipelineState {
  kCreated = 0,
  kInitialized = 1,
  kRunning = 2,
  kFinished = 3,
  kDeinitialized = 4,
};

using ElementNameList = std::vector<std::string>;

}  // namespace orinmllm::graph
