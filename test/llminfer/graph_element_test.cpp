#include "graph/element/element_manager.h"
#include "graph/node/node_base.h"
#include "test/llminfer/test_utils.h"

#include <string>
#include <vector>

int main() {
  using orinmllm::graph::Element;
  using orinmllm::graph::ElementManager;
  using orinmllm::graph::LambdaNode;

  ElementManager manager;
  LambdaNode* first = nullptr;
  LambdaNode* second = nullptr;
  LambdaNode* third = nullptr;
  EXPECT_TRUE_OR_EXIT(manager.create<LambdaNode>("first", &first));
  EXPECT_TRUE_OR_EXIT(manager.create<LambdaNode>("second", &second));
  EXPECT_TRUE_OR_EXIT(manager.create<LambdaNode>("third", &third));
  EXPECT_TRUE_OR_EXIT(manager.add_depend("second", "first"));
  EXPECT_TRUE_OR_EXIT(manager.add_depend("third", "second"));
  EXPECT_TRUE_OR_EXIT(third->has_depend(second));

  std::vector<Element*> order;
  EXPECT_TRUE_OR_EXIT(manager.topo_sort(&order));
  EXPECT_EQ_OR_EXIT(order.size(), static_cast<std::size_t>(3));
  EXPECT_TRUE_OR_EXIT(order[0] == first);
  EXPECT_TRUE_OR_EXIT(order[1] == second);
  EXPECT_TRUE_OR_EXIT(order[2] == third);
  EXPECT_TRUE_OR_EXIT(!manager.has_cycle());
  return EXIT_SUCCESS;
}