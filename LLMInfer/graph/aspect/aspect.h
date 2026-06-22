#pragma once

#include "graph/aspect/aspect_object.h"

namespace orinmllm::graph {

class Aspect : public AspectObject {
 public:
  Aspect() = default;
  ~Aspect() override = default;

  virtual bool begin_init() { return true; }
  virtual bool finish_init() { return true; }
  virtual bool begin_run() { return true; }
  virtual bool finish_run(const bool is_run_success) {
    (void)is_run_success;
    return true;
  }
  virtual bool begin_deinit() { return true; }
  virtual bool finish_deinit() { return true; }
};

}  // namespace orinmllm::graph

