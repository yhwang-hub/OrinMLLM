#include "stream/stream.h"
#include "test/llminfer/test_utils.h"

#include <cstdint>

int main() {
  using orinmllm::stream::EventHandle;
  using orinmllm::stream::Stream;
  using orinmllm::stream::StreamManager;

  Stream stream;
  EXPECT_TRUE_OR_EXIT(stream.valid());
  bool is_ready = false;
  EXPECT_TRUE_OR_EXIT(stream.query(&is_ready));
  EXPECT_TRUE_OR_EXIT(is_ready);
  EXPECT_TRUE_OR_EXIT(stream.synchronize());
  EventHandle fake_event = reinterpret_cast<EventHandle>(static_cast<uintptr_t>(1));
  EXPECT_TRUE_OR_EXIT(stream.wait_event(fake_event));

  StreamManager manager;
  Stream* managed = nullptr;
  EXPECT_TRUE_OR_EXIT(manager.create(3, 0, &managed));
  EXPECT_TRUE_OR_EXIT(managed != nullptr);
  EXPECT_EQ_OR_EXIT(manager.size(), static_cast<std::size_t>(1));
  EXPECT_TRUE_OR_EXIT(manager.synchronize_all());
  EXPECT_TRUE_OR_EXIT(manager.erase(3));
  EXPECT_EQ_OR_EXIT(manager.size(), static_cast<std::size_t>(0));
  return EXIT_SUCCESS;
}