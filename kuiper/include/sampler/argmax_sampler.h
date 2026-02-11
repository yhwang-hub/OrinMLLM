//
// Created by fss on 24-6-9.
//

#ifndef LLAMA_INFER_NON_SAMPLER_H
#define LLAMA_INFER_NON_SAMPLER_H
#include <base/base.h>
#include "sampler.h"
namespace sampler {
class ArgmaxSampler : public Sampler {
 public:
  explicit ArgmaxSampler(base::DeviceType device_type) : Sampler(device_type) {}

  size_t sample(const float* logits, size_t size, void* stream) override;
  
  // Optimized sample using pre-allocated GPU and pinned buffers
  // This avoids per-call memory allocation and enables true async D2H transfer
  void sample_prealloc(const float* logits, size_t size, 
                       size_t* output_gpu, size_t* output_pinned, void* stream);
};
}  // namespace sampler
#endif  // LLAMA_INFER_NON_SAMPLER_H
