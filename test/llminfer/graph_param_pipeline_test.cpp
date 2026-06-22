#include "graph/node/node_base.h"
#include "graph/param/param.h"
#include "graph/pipeline/pipeline_factory.h"
#include "test/llminfer/test_utils.h"

#include <string>

class CounterParam : public orinmllm::graph::Param {
 public:
  bool reset() override {
    value_ = 0;
    return true;
  }

  int value() const { return value_; }
  bool set_value(const int value) {
    value_ = value;
    return true;
  }

 private:
  int value_ = 0;
};

int main() {
  using orinmllm::graph::LambdaNode;
  using orinmllm::graph::PipelineFactory;

  auto pipeline = PipelineFactory::create("param_pipeline");
  EXPECT_TRUE_OR_EXIT(pipeline != nullptr);
  EXPECT_TRUE_OR_EXIT(pipeline->create_param<CounterParam>("counter", true));
  CounterParam* const param = pipeline->get_param<CounterParam>("counter");
  EXPECT_TRUE_OR_EXIT(param != nullptr);
  EXPECT_TRUE_OR_EXIT(param->add_backtrace("pipeline"));

  LambdaNode* producer = nullptr;
  LambdaNode* consumer = nullptr;
  EXPECT_TRUE_OR_EXIT(pipeline->create<LambdaNode>("producer", &producer));
  EXPECT_TRUE_OR_EXIT(pipeline->create<LambdaNode>("consumer", &consumer));
  EXPECT_TRUE_OR_EXIT(producer->set_process_func([param]() {
    return param->set_value(7);
  }));
  EXPECT_TRUE_OR_EXIT(consumer->set_process_func([param]() {
    return param->value() == 7;
  }));
  EXPECT_TRUE_OR_EXIT(pipeline->add_depend("consumer", "producer"));
  EXPECT_TRUE_OR_EXIT(pipeline->run());
  std::string dump_text;
  EXPECT_TRUE_OR_EXIT(pipeline->dump(&dump_text));
  EXPECT_TRUE_OR_EXIT(dump_text.find("producer") != std::string::npos);
  EXPECT_TRUE_OR_EXIT(dump_text.find("consumer") != std::string::npos);
  EXPECT_TRUE_OR_EXIT(pipeline->deinit());
  EXPECT_EQ_OR_EXIT(param->value(), 0);
  return EXIT_SUCCESS;
}