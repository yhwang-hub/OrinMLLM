#include "graph/aspect/template_aspect.h"
#include "graph/node/node_base.h"
#include "test/llminfer/test_utils.h"

#include <memory>
#include <vector>

int main() {
  using orinmllm::graph::AspectType;
  using orinmllm::graph::LambdaNode;
  using orinmllm::graph::TemplateAspect;

  LambdaNode node;
  std::vector<int> calls;
  auto aspect = std::make_unique<TemplateAspect>();
  EXPECT_TRUE_OR_EXIT(aspect->set_begin_init_hook([&calls]() {
    calls.push_back(1);
    return true;
  }));
  EXPECT_TRUE_OR_EXIT(aspect->set_finish_run_hook([&calls](const bool is_run_success) {
    if (is_run_success) {
      calls.push_back(2);
    }
    return true;
  }));
  EXPECT_TRUE_OR_EXIT(node.aspect_manager()->add_aspect(std::move(aspect), &node));
  EXPECT_TRUE_OR_EXIT(node.aspect_manager()->trigger(AspectType::kBeginInit));
  EXPECT_TRUE_OR_EXIT(node.aspect_manager()->trigger(AspectType::kFinishRun, true));
  EXPECT_EQ_OR_EXIT(calls.size(), static_cast<std::size_t>(2));
  EXPECT_EQ_OR_EXIT(calls[0], 1);
  EXPECT_EQ_OR_EXIT(calls[1], 2);
  return EXIT_SUCCESS;
}