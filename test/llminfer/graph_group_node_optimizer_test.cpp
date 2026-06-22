#include "graph/group/group.h"
#include "graph/node/node_base.h"
#include "graph/optimizer/separate.h"
#include "graph/optimizer/trim.h"
#include "test/llminfer/test_utils.h"

#include <vector>

int main() {
  using orinmllm::graph::ElementManager;
  using orinmllm::graph::Group;
  using orinmllm::graph::LambdaNode;
  using orinmllm::graph::SeparateOptimizer;
  using orinmllm::graph::TrimOptimizer;

  int value = 0;
  Group group;
  LambdaNode* add_one = nullptr;
  LambdaNode* multiply = nullptr;
  EXPECT_TRUE_OR_EXIT(group.create_inner<LambdaNode>("add_one", &add_one));
  EXPECT_TRUE_OR_EXIT(group.create_inner<LambdaNode>("multiply", &multiply));
  EXPECT_TRUE_OR_EXIT(add_one->set_process_func([&value]() {
    ++value;
    return true;
  }));
  EXPECT_TRUE_OR_EXIT(multiply->set_process_func([&value]() {
    value *= 3;
    return true;
  }));
  EXPECT_TRUE_OR_EXIT(group.add_inner_depend("multiply", "add_one"));
  EXPECT_TRUE_OR_EXIT(group.init());
  EXPECT_TRUE_OR_EXIT(group.process());
  EXPECT_TRUE_OR_EXIT(group.deinit());
  EXPECT_EQ_OR_EXIT(value, 3);

  ElementManager manager;
  LambdaNode* a = nullptr;
  LambdaNode* b = nullptr;
  LambdaNode* c = nullptr;
  EXPECT_TRUE_OR_EXIT(manager.create<LambdaNode>("a", &a));
  EXPECT_TRUE_OR_EXIT(manager.create<LambdaNode>("b", &b));
  EXPECT_TRUE_OR_EXIT(manager.create<LambdaNode>("c", &c));
  EXPECT_TRUE_OR_EXIT(manager.add_depend("b", "a"));
  TrimOptimizer trim;
  EXPECT_TRUE_OR_EXIT(trim.optimize(&manager));
  SeparateOptimizer separate;
  EXPECT_TRUE_OR_EXIT(separate.optimize(&manager));
  EXPECT_EQ_OR_EXIT(separate.components().size(), static_cast<std::size_t>(2));
  return EXIT_SUCCESS;
}