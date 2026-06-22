#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace orinmllm::stream {

using StreamHandle = void*;
using EventHandle = void*;

struct StreamBackend {
	std::function<bool(const uint32_t flags, StreamHandle* const out_handle)> create;
	std::function<bool(StreamHandle handle)> destroy;
	std::function<bool(StreamHandle handle)> synchronize;
	std::function<bool(StreamHandle handle, bool* const out_ready)> query;
	std::function<bool(StreamHandle stream, EventHandle event, const uint32_t flags)> wait_event;
};

class Stream {
 public:
	explicit Stream(const uint32_t flags = 0);
	~Stream();

	Stream(const Stream&) = delete;
	Stream& operator=(const Stream&) = delete;
	Stream(Stream&& other) noexcept;
	Stream& operator=(Stream&& other) noexcept;

	bool synchronize() const;
	bool query(bool* const out_ready) const;
	bool wait_event(EventHandle event, const uint32_t flags = 0) const;
	StreamHandle handle() const;
	bool valid() const;

	static bool set_backend(const StreamBackend& backend);
	static bool reset_backend();

 private:
	bool release();

	StreamHandle handle_ = nullptr;
};

class StreamManager {
 public:
	StreamManager() = default;
	~StreamManager() = default;

	bool create(const int32_t id, const uint32_t flags, Stream** const out_stream);
	Stream* get(const int32_t id) const;
	bool synchronize_all() const;
	bool erase(const int32_t id);
	void clear();
	std::size_t size() const;

 private:
	mutable std::mutex mutex_;
	std::unordered_map<int32_t, std::unique_ptr<Stream>> streams_;
};

}  // namespace orinmllm::stream
