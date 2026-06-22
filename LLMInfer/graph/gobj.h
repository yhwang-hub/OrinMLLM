#pragma once

namespace orinmllm::graph {

class GObj {
 public:
  GObj() = default;
  virtual ~GObj() = default;

  virtual bool init() { return true; }
  virtual bool run() = 0;
  virtual bool deinit() { return true; }
};

class ParamObj : public GObj {
 public:
  ParamObj() = default;
  ~ParamObj() override = default;

  bool run() override { return false; }
};

class Runnable : public GObj {
 public:
  Runnable() = default;
  ~Runnable() override = default;
};

}  // namespace orinmllm::graph
