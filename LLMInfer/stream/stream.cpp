#include "stream/stream.h"

#include <atomic>

namespace orinmllm::stream {
namespace {

std::atomic<uint64_t> g_cpu_stream_id{1};
std::mutex g_backend_mutex;

bool CpuCreate(const uint32_t flags, StreamHandle* const out_handle) {
	(void)flags;
	if (out_handle == nullptr) {
		return false;
	}
	const uint64_t id = g_cpu_stream_id.fetch_add(1, std::memory_order_relaxed);
	*out_handle = reinterpret_cast<StreamHandle>(id);
	return true;
}

bool CpuDestroy(StreamHandle handle) { return handle != nullptr; }

bool CpuSynchronize(StreamHandle handle) { return handle != nullptr; }

bool CpuQuery(StreamHandle handle, bool* const out_ready) {
	if (handle == nullptr || out_ready == nullptr) {
		return false;
	}
	*out_ready = true;
	return true;
}

bool CpuWaitEvent(StreamHandle stream, EventHandle event, const uint32_t flags) {
	(void)flags;
	return stream != nullptr && event != nullptr;
}

StreamBackend DefaultBackend() {
	StreamBackend backend;
	backend.create = CpuCreate;
	backend.destroy = CpuDestroy;
	backend.synchronize = CpuSynchronize;
	backend.query = CpuQuery;
	backend.wait_event = CpuWaitEvent;
	return backend;
}

StreamBackend& ActiveBackend() {
	static StreamBackend backend = DefaultBackend();
	return backend;
}

}  // namespace

Stream::Stream(const uint32_t flags) {
	std::lock_guard<std::mutex> lock(g_backend_mutex);
	if (ActiveBackend().create != nullptr) {
		(void)ActiveBackend().create(flags, &handle_);
	}
}

Stream::~Stream() { (void)release(); }

Stream::Stream(Stream&& other) noexcept : handle_(other.handle_) { other.handle_ = nullptr; }

Stream& Stream::operator=(Stream&& other) noexcept {
	if (this != &other) {
		(void)release();
		handle_ = other.handle_;
		other.handle_ = nullptr;
	}
	return *this;
}

bool Stream::synchronize() const {
	std::lock_guard<std::mutex> lock(g_backend_mutex);
	return handle_ != nullptr && ActiveBackend().synchronize != nullptr &&
				 ActiveBackend().synchronize(handle_);
}

bool Stream::query(bool* const out_ready) const {
	if (out_ready == nullptr) {
		return false;
	}
	std::lock_guard<std::mutex> lock(g_backend_mutex);
	return handle_ != nullptr && ActiveBackend().query != nullptr &&
				 ActiveBackend().query(handle_, out_ready);
}

bool Stream::wait_event(EventHandle event, const uint32_t flags) const {
	std::lock_guard<std::mutex> lock(g_backend_mutex);
	return handle_ != nullptr && event != nullptr && ActiveBackend().wait_event != nullptr &&
				 ActiveBackend().wait_event(handle_, event, flags);
}

StreamHandle Stream::handle() const { return handle_; }

bool Stream::valid() const { return handle_ != nullptr; }

bool Stream::set_backend(const StreamBackend& backend) {
	if (backend.create == nullptr || backend.destroy == nullptr || backend.synchronize == nullptr ||
			backend.query == nullptr || backend.wait_event == nullptr) {
		return false;
	}
	std::lock_guard<std::mutex> lock(g_backend_mutex);
	ActiveBackend() = backend;
	return true;
}

bool Stream::reset_backend() {
	std::lock_guard<std::mutex> lock(g_backend_mutex);
	ActiveBackend() = DefaultBackend();
	return true;
}

bool Stream::release() {
	if (handle_ == nullptr) {
		return true;
	}
	std::lock_guard<std::mutex> lock(g_backend_mutex);
	const bool is_success = ActiveBackend().destroy != nullptr && ActiveBackend().destroy(handle_);
	handle_ = nullptr;
	return is_success;
}

bool StreamManager::create(const int32_t id, const uint32_t flags, Stream** const out_stream) {
	if (out_stream == nullptr) {
		return false;
	}
	std::lock_guard<std::mutex> lock(mutex_);
	if (streams_.find(id) != streams_.end()) {
		return false;
	}
	auto stream = std::make_unique<Stream>(flags);
	if (!stream->valid()) {
		return false;
	}
	*out_stream = stream.get();
	streams_.emplace(id, std::move(stream));
	return true;
}

Stream* StreamManager::get(const int32_t id) const {
	std::lock_guard<std::mutex> lock(mutex_);
	const auto iter = streams_.find(id);
	return iter == streams_.end() ? nullptr : iter->second.get();
}

bool StreamManager::synchronize_all() const {
	std::lock_guard<std::mutex> lock(mutex_);
	for (const auto& item : streams_) {
		if (item.second == nullptr || !item.second->synchronize()) {
			return false;
		}
	}
	return true;
}

bool StreamManager::erase(const int32_t id) {
	std::lock_guard<std::mutex> lock(mutex_);
	return streams_.erase(id) > 0;
}

void StreamManager::clear() {
	std::lock_guard<std::mutex> lock(mutex_);
	streams_.clear();
}

std::size_t StreamManager::size() const {
	std::lock_guard<std::mutex> lock(mutex_);
	return streams_.size();
}

}  // namespace orinmllm::stream
