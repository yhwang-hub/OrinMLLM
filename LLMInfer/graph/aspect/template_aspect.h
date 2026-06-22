#pragma once

#include <functional>

#include "graph/aspect/aspect.h"

namespace orinmllm::graph {

class TemplateAspect : public Aspect {
 public:
	using Hook = std::function<bool()>;
	using FinishRunHook = std::function<bool(bool)>;

	TemplateAspect() = default;
	~TemplateAspect() override = default;

	bool begin_init() override;
	bool finish_init() override;
	bool begin_run() override;
	bool finish_run(const bool is_run_success) override;
	bool begin_deinit() override;
	bool finish_deinit() override;

	bool set_begin_init_hook(const Hook& hook);
	bool set_finish_init_hook(const Hook& hook);
	bool set_begin_run_hook(const Hook& hook);
	bool set_finish_run_hook(const FinishRunHook& hook);
	bool set_begin_deinit_hook(const Hook& hook);
	bool set_finish_deinit_hook(const Hook& hook);

 private:
	Hook begin_init_hook_;
	Hook finish_init_hook_;
	Hook begin_run_hook_;
	FinishRunHook finish_run_hook_;
	Hook begin_deinit_hook_;
	Hook finish_deinit_hook_;
};

inline bool TemplateAspect::begin_init() {
	return begin_init_hook_ == nullptr || begin_init_hook_();
}

inline bool TemplateAspect::finish_init() {
	return finish_init_hook_ == nullptr || finish_init_hook_();
}

inline bool TemplateAspect::begin_run() {
	return begin_run_hook_ == nullptr || begin_run_hook_();
}

inline bool TemplateAspect::finish_run(const bool is_run_success) {
	return finish_run_hook_ == nullptr || finish_run_hook_(is_run_success);
}

inline bool TemplateAspect::begin_deinit() {
	return begin_deinit_hook_ == nullptr || begin_deinit_hook_();
}

inline bool TemplateAspect::finish_deinit() {
	return finish_deinit_hook_ == nullptr || finish_deinit_hook_();
}

inline bool TemplateAspect::set_begin_init_hook(const Hook& hook) {
	begin_init_hook_ = hook;
	return true;
}

inline bool TemplateAspect::set_finish_init_hook(const Hook& hook) {
	finish_init_hook_ = hook;
	return true;
}

inline bool TemplateAspect::set_begin_run_hook(const Hook& hook) {
	begin_run_hook_ = hook;
	return true;
}

inline bool TemplateAspect::set_finish_run_hook(const FinishRunHook& hook) {
	finish_run_hook_ = hook;
	return true;
}

inline bool TemplateAspect::set_begin_deinit_hook(const Hook& hook) {
	begin_deinit_hook_ = hook;
	return true;
}

inline bool TemplateAspect::set_finish_deinit_hook(const Hook& hook) {
	finish_deinit_hook_ = hook;
	return true;
}

}  // namespace orinmllm::graph
