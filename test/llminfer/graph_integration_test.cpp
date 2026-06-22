#include "graph/aspect/template_aspect.h"
#include "graph/node/node_base.h"
#include "graph/pipeline/pipeline_factory.h"
#include "test/llminfer/test_utils.h"

#include <memory>
#include <string>
#include <vector>

int main() {
  using orinmllm::graph::LambdaNode;
  using orinmllm::graph::PipelineFactory;
  using orinmllm::graph::TemplateAspect;

  auto pipeline = PipelineFactory::create("dag_integration");
  LambdaNode* load = nullptr;
  LambdaNode* decode = nullptr;
  LambdaNode* sample = nullptr;
  std::vector<std::string> events;
  int value = 0;
  EXPECT_TRUE_OR_EXIT(pipeline->create<LambdaNode>("load", &load));
  EXPECT_TRUE_OR_EXIT(pipeline->create<LambdaNode>("decode", &decode));
  EXPECT_TRUE_OR_EXIT(pipeline->create<LambdaNode>("sample", &sample));
  EXPECT_TRUE_OR_EXIT(load->set_process_func([&value]() {
    value = 2;
    return true;
  }));
  EXPECT_TRUE_OR_EXIT(decode->set_process_func([&value]() {
    value += 5;
    return true;
  }));
  EXPECT_TRUE_OR_EXIT(sample->set_process_func([&value]() { return value == 7; }));
  EXPECT_TRUE_OR_EXIT(pipeline->add_depend("decode", "load"));
  EXPECT_TRUE_OR_EXIT(pipeline->add_depend("sample", "decode"));

  auto aspect = std::make_unique<TemplateAspect>();
  EXPECT_TRUE_OR_EXIT(aspect->set_begin_run_hook([&events]() {
    events.push_back("begin");
    return true;
  }));
  EXPECT_TRUE_OR_EXIT(aspect->set_finish_run_hook([&events](const bool is_run_success) {
    events.push_back(is_run_success ? "ok" : "fail");
    return true;
  }));
  EXPECT_TRUE_OR_EXIT(decode->aspect_manager()->add_aspect(std::move(aspect), decode));
  EXPECT_TRUE_OR_EXIT(pipeline->run());
  EXPECT_EQ_OR_EXIT(value, 7);
  EXPECT_EQ_OR_EXIT(events.size(), static_cast<std::size_t>(2));
  std::string dump_text;
  EXPECT_TRUE_OR_EXIT(pipeline->dump(&dump_text));
  EXPECT_TRUE_OR_EXIT(dump_text.find("sample <- [decode]") != std::string::npos);
  EXPECT_TRUE_OR_EXIT(pipeline->deinit());
  return EXIT_SUCCESS;
}