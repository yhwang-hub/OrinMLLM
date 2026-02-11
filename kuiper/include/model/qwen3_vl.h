#ifndef KUIPER_INCLUDE_MODEL_QWEN3_VL_H_
#define KUIPER_INCLUDE_MODEL_QWEN3_VL_H_
#include <base/cuda_config.h>
#include "model.h"
#include "qwen3.h"
#include "op/add.h"
#include "op/embedding.h"
#include "op/rope.h"
#include "op/swiglu.h"
#include "op/matmul.h"
#include "op/vision_layers.h"

namespace model {

/**
 * @brief Qwen3-VL Vision Config
 * 
 * Vision encoder configuration for Qwen3-VL-8B:
 * - hidden_size: 1152
 * - intermediate_size: 4304  
 * - num_heads: 16
 * - depth: 27 transformer blocks
 * - patch_size: 16x16
 * - temporal_patch_size: 2 (for video)
 * - spatial_merge_size: 2 (4 patches merged to 1)
 * - out_hidden_size: 4096 (same as LLM dim)
 * - num_position_embeddings: 2304 (48x48 grid)
 * - deepstack_visual_indexes: [8, 16, 24]
 */
struct Qwen3VLVisionConfig {
  int32_t hidden_size = 1152;
  int32_t intermediate_size = 4304;
  int32_t num_heads = 16;
  int32_t depth = 27;
  int32_t patch_size = 16;
  int32_t temporal_patch_size = 2;
  int32_t in_channels = 3;
  int32_t spatial_merge_size = 2;
  int32_t out_hidden_size = 4096;
  int32_t num_position_embeddings = 2304;
  std::vector<int32_t> deepstack_visual_indexes = {8, 16, 24};
  
  int32_t head_dim() const { return hidden_size / num_heads; }
  int32_t num_deepstack_mergers() const { return deepstack_visual_indexes.size(); }
};

/**
 * @brief Qwen3-VL Text/LLM Config
 */
struct Qwen3VLTextConfig {
  int32_t hidden_size = 4096;
  int32_t intermediate_size = 12288;
  int32_t num_hidden_layers = 36;
  int32_t num_attention_heads = 32;
  int32_t num_key_value_heads = 8;
  int32_t vocab_size = 151936;
  int32_t max_position_embeddings = 262144;
  int32_t head_dim = 128;
  float rms_norm_eps = 1e-6f;
  float rope_theta = 5000000.0f;
  
  // M-RoPE configuration for multimodal position encoding
  // mrope_section defines how head_dim is split for 3D positions (temporal, height, width)
  // For Qwen3-VL: [24, 20, 20] means 24 pairs for t, 20 for h, 20 for w
  // Total: 24+20+20 = 64 pairs = 128 dimensions (head_dim)
  std::vector<int32_t> mrope_section = {24, 20, 20};
};

/**
 * @brief Special tokens for Qwen3-VL
 */
struct Qwen3VLSpecialTokens {
  int32_t image_token_id = 151655;
  int32_t video_token_id = 151656;
  int32_t vision_start_token_id = 151652;
  int32_t vision_end_token_id = 151653;
  int32_t eos_token_id = 151645;
};

/**
 * @brief Full model config for Qwen3-VL
 */
struct Qwen3VLConfig {
  Qwen3VLVisionConfig vision;
  Qwen3VLTextConfig text;
  Qwen3VLSpecialTokens special_tokens;
  bool has_lm_head = true;
};

/**
 * @brief Image preprocessing result
 */
struct ImageData {
  tensor::Tensor pixel_values;  // [num_patches, patch_dim] in FP16
  int32_t grid_h = 0;           // Number of patches in height
  int32_t grid_w = 0;           // Number of patches in width
  int32_t grid_t = 1;           // Number of temporal frames (1 for image)
  int32_t num_patches = 0;      // Total number of patches
  int32_t num_vision_tokens = 0; // After spatial merge
};

/**
 * @brief Vision encoder workspace buffers (pre-allocated for performance)
 * These buffers are reused across all transformer blocks to avoid repeated allocation
 */
struct VisionWorkspace {
  int max_patches = 0;  // Maximum number of patches this workspace can handle
  
  // Transformer block intermediate buffers
  tensor::Tensor normed1;           // [max_patches, hidden_size]
  tensor::Tensor qkv;               // [max_patches, 3*hidden_size]
  tensor::Tensor query;             // [max_patches, hidden_size]
  tensor::Tensor key;               // [max_patches, hidden_size]
  tensor::Tensor value;             // [max_patches, hidden_size]
  tensor::Tensor attn_out;          // [max_patches, hidden_size]
  tensor::Tensor normed2;           // [max_patches, hidden_size]
  tensor::Tensor mlp_intermediate;  // [max_patches, intermediate_size]
  tensor::Tensor proj_out;          // [max_patches, hidden_size]
  tensor::Tensor output;            // [max_patches, hidden_size] - double buffer 0
  tensor::Tensor output2;           // [max_patches, hidden_size] - double buffer 1
  
  // Attention workspace (to avoid allocation in flash_attention_cu)
  tensor::Tensor q_transposed;      // [num_heads, max_patches, head_dim]
  tensor::Tensor k_transposed;      // [num_heads, max_patches, head_dim]
  tensor::Tensor v_transposed;      // [num_heads, max_patches, head_dim]
  tensor::Tensor out_transposed;    // [num_heads, max_patches, head_dim]
  tensor::Tensor attn_scores;       // [num_heads, max_patches, max_patches]
  
  // Check if workspace is valid for given number of patches
  bool is_valid_for(int num_patches) const {
    return max_patches >= num_patches && max_patches > 0;
  }
};

/**
 * @brief Vision Encoder layers
 */
struct Qwen3VLVisionLayers {
  // Patch embedding (Conv3d)
  tensor::Tensor patch_embed_weight;  // [hidden_size, 3, temporal_patch_size, patch_size, patch_size]
  tensor::Tensor patch_embed_bias;    // [hidden_size]
  
  // Position embedding
  tensor::Tensor pos_embed_weight;    // [num_position_embeddings, hidden_size]
  
  // Transformer blocks (per layer)
  struct Block {
    tensor::Tensor norm1_weight;      // [hidden_size]
    tensor::Tensor norm1_bias;        // [hidden_size]
    tensor::Tensor norm2_weight;      // [hidden_size]
    tensor::Tensor norm2_bias;        // [hidden_size]
    tensor::Tensor qkv_weight;        // [3*hidden_size, hidden_size]
    tensor::Tensor qkv_bias;          // [3*hidden_size]
    tensor::Tensor proj_weight;       // [hidden_size, hidden_size]
    tensor::Tensor proj_bias;         // [hidden_size]
    tensor::Tensor mlp_fc1_weight;    // [intermediate_size, hidden_size]
    tensor::Tensor mlp_fc1_bias;      // [intermediate_size]
    tensor::Tensor mlp_fc2_weight;    // [hidden_size, intermediate_size]
    tensor::Tensor mlp_fc2_bias;      // [hidden_size]
  };
  std::vector<Block> blocks;
  
  // Main merger (vision to LLM projection)
  struct Merger {
    tensor::Tensor norm_weight;       // [hidden_size]
    tensor::Tensor norm_bias;         // [hidden_size]
    tensor::Tensor fc1_weight;        // [merged_hidden, merged_hidden] where merged_hidden = hidden_size * spatial_merge_size^2
    tensor::Tensor fc1_bias;          // [merged_hidden]
    tensor::Tensor fc2_weight;        // [out_hidden_size, merged_hidden]
    tensor::Tensor fc2_bias;          // [out_hidden_size]
  };
  Merger merger;
  std::vector<Merger> deepstack_mergers;  // 3 additional mergers
  
  void to_cuda(cudaStream_t stream);
};

/**
 * @brief Vision-specific operation layers for Qwen3-VL
 * These wrap kernel calls as layer->forward() for unified access
 */
struct VisionVLLayers {
  std::shared_ptr<op::ExtractPatchesLayer> extract_patches_layer_;
  std::shared_ptr<op::BiasAddResidualLayer> bias_add_residual_layer_;
  std::shared_ptr<op::PosEmbedInterpolateLayer> pos_embed_interpolate_layer_;
  std::shared_ptr<op::LayerNormWithBiasLayer> layernorm_with_bias_layer_;
  std::shared_ptr<op::FusedSplitRopeTransposeLayer> fused_split_rope_transpose_layer_;
  std::shared_ptr<op::VisionAttentionLayer> vision_attention_layer_;
  std::shared_ptr<op::VisionMLPLayer> vision_mlp_layer_;
  std::shared_ptr<op::SpatialMergeLayer> spatial_merge_layer_;
  std::shared_ptr<op::VisionMergerMLPLayer> vision_merger_mlp_layer_;
  std::shared_ptr<op::FusedMultimodalEmbedLayer> fused_multimodal_embed_layer_;
};

/**
 * @brief Qwen3-VL Model (Vision-Language Model)
 * 
 * Model architecture:
 * 1. Vision Encoder (ViT): Processes images into visual tokens
 *    - Patch embedding: Conv3d to convert image patches to embeddings
 *    - Position embedding: Learnable position embeddings
 *    - 27 Transformer blocks with self-attention and MLP
 *    - Merger: Projects vision features to LLM dimension
 *    - Deepstack: Additional features from intermediate layers
 * 
 * 2. Language Model (Qwen3): Standard causal LLM
 *    - 36 transformer layers
 *    - RMSNorm, rotary position embeddings
 *    - Grouped-query attention with q_norm/k_norm
 * 
 * 3. Multimodal fusion:
 *    - Replace <image> tokens with visual embeddings
 *    - Use M-RoPE for position encoding
 */
class Qwen3VLModel : public Model {
public:
  explicit Qwen3VLModel(base::TokenizerType tokenizer_type, 
                        std::string token_path,
                        std::string model_path);
  
  ~Qwen3VLModel();

  base::Status init(base::DeviceType device_type) override;

  base::Status predict(const tensor::Tensor& input, const tensor::Tensor& pos_tensor,
                       bool is_prompt, int& next) const override;

  base::Status forward(const tensor::Tensor& input, const tensor::Tensor& pos_tensor,
                       int& next) const override;

  op::EmbeddingOutput embedding(const std::vector<int>& tokens) const override;

  // Vision-Language specific methods
  
  /**
   * @brief Preprocess an image for vision encoder
   * @param image_path Path to the image file
   * @param max_pixels Maximum pixels for image resize (default: 1003520 = 14*14*4*1280)
   *                   Lower values = faster ViT but reduced image quality
   *                   Suggested values: 1003520 (default), 500000, 400000, 300000
   * @return ImageData with preprocessed pixel values
   */
  ImageData preprocess_image(const std::string& image_path, int max_pixels = 1003520) const;
  
  /**
   * @brief Run vision encoder to get visual embeddings
   * @param image_data Preprocessed image data
   * @return Visual embeddings [num_vision_tokens, out_hidden_size * (1 + num_deepstack)]
   */
  tensor::Tensor encode_image(const ImageData& image_data) const;
  
  /**
   * @brief Prepare input embeddings for multimodal prefill
   * @param tokens Input token ids
   * @param image_data Preprocessed image data (optional)
   * @return Combined text and visual embeddings
   */
  tensor::Tensor prepare_multimodal_embeddings(
      const std::vector<int>& tokens,
      const ImageData* image_data = nullptr) const;
  
  /**
   * @brief Prefill with multimodal input
   * @param tokens Input token ids
   * @param image_path Path to image (optional)
   */
  base::Status multimodal_prefill(const std::vector<int>& tokens,
                                   const std::string& image_path = "") const;
  
  /**
   * @brief Prefill phase with embeddings
   */
  base::Status prefill(const tensor::Tensor& input_embeddings, 
                       int32_t seq_len, int32_t start_pos) const;
  
  /**
   * @brief Single-token decode step
   */
  base::Status decode_step(const tensor::Tensor& input, int32_t pos, int& next) const;
  
  /**
   * @brief Optimized decode step that assumes embedding is already in decode_input buffer
   * This avoids D2D copy by using embedding_to_decode_input() beforehand
   */
  base::Status decode_step_optimized(int32_t pos, int& next) const;
  
  /**
   * @brief Embed single token directly into decode_input buffer
   * Avoids the D2D copy that would otherwise be needed in decode_step
   */
  void embedding_to_decode_input(int token_id) const;
  
  /**
   * @brief Sample the first token after prefill
   * Uses the hidden state from the last token in prefill
   */
  int sample_first_token() const;
  
  /**
   * @brief Generate response for image + text input
   */
  std::string generate(const std::string& image_path,
                       const std::string& prompt,
                       int max_tokens = 256) const;
  
  // Configuration access
  const Qwen3VLConfig& get_vl_config() const { return vl_config_; }
  
  // CUDA Graph management
  void enable_cuda_graph(bool enable);
  bool is_cuda_graph_enabled() const;
  void invalidate_cuda_graph();
  
  // Clear KV cache
  void clear_kv_cache();
  
  std::shared_ptr<kernel::CudaConfig> get_cuda_config() const { return cuda_config_; }

private:
  // Model initialization
  base::Status load_vl_model_file();
  void init_mem() override;
  base::Status create_layers() override;
  void create_param_layers() override;
  void create_nonparam_layers() override;
  void create_param_quant_layers() override;
  
  // Vision encoder forward pass
  tensor::Tensor vision_patch_embed(const ImageData& image_data) const;
  tensor::Tensor vision_add_pos_embed(const tensor::Tensor& patch_embeds, 
                                       int grid_h, int grid_w) const;
  // vision_transformer_block with double-buffering: output is written to output_buffer,
  // and hidden_states is used as residual (no copy needed if different buffers)
  void vision_transformer_block(const tensor::Tensor& hidden_states,
                                 tensor::Tensor& output_buffer,
                                 int block_idx,
                                 const tensor::Tensor& cu_seqlens,
                                 int max_seqlen,
                                 const tensor::Tensor& cos_cache,
                                 const tensor::Tensor& sin_cache,
                                 VisionWorkspace& workspace) const;
  tensor::Tensor vision_merger(const tensor::Tensor& hidden_states,
                               int grid_h, int grid_w, int grid_t,
                               bool is_deepstack, int merger_idx = 0) const;
  
  // Compute rotary position embeddings for vision encoder
  std::pair<tensor::Tensor, tensor::Tensor> compute_vision_rotary_emb(
      int grid_h, int grid_w, int grid_t) const;
  
  // LLM forward pass (inherited from Qwen3 structure)
  void attention_rms(int32_t layer_idx, const tensor::Tensor& input) const;
  void attention_qkv(int32_t layer_idx, const tensor::Tensor& pos_tensor) const;
  void attention_mha(int32_t layer_idx, const tensor::Tensor& pos_tensor) const;
  void feed_forward(int32_t layer_idx, const tensor::Tensor& input) const;
  void cls_logits(const tensor::Tensor& input) const;
  
  // CUDA Graph compatible versions (use GPU-resident position tensors)
  // rope_pos_gpu: M-RoPE text position for computing rotary embeddings
  // kv_cache_pos_gpu: KV cache position for indexing into KV cache
  void attention_qkv_with_graph(int32_t layer_idx, 
                                 const tensor::Tensor& rope_pos_gpu,
                                 const tensor::Tensor& kv_cache_pos_gpu) const;
  void attention_mha_with_graph(int32_t layer_idx, const tensor::Tensor& kv_cache_pos_gpu) const;
  
  // Batched operations for prefill
  void batched_attention_rms(int32_t layer_idx, const tensor::Tensor& input, 
                             const tensor::Tensor& output, int32_t seq_len) const;
  void batched_attention_qkv(int32_t layer_idx, const tensor::Tensor& rms_out,
                             const tensor::Tensor& query_out, const tensor::Tensor& key_out, 
                             const tensor::Tensor& value_out,
                             int32_t seq_len, int32_t start_pos) const;
  void batched_attention_mha(int32_t layer_idx, const tensor::Tensor& query,
                             tensor::Tensor& mha_out, int32_t seq_len, int32_t start_pos) const;
  void batched_feed_forward(int32_t layer_idx, const tensor::Tensor& input, int32_t seq_len) const;
  void batched_feed_forward_optimized(int32_t layer_idx, const tensor::Tensor& input,
                                      tensor::Tensor& ffn_norm_out,
                                      tensor::Tensor& w1_out, tensor::Tensor& w3_out,
                                      tensor::Tensor& w2_out, int32_t seq_len) const;
  
  int32_t post_processing(const tensor::Tensor& pos, bool is_prompt) const override;

private:
  Qwen3VLConfig vl_config_;
  std::shared_ptr<kernel::CudaConfig> cuda_config_;
  
  // Vision encoder layers
  std::unique_ptr<Qwen3VLVisionLayers> vision_layers_;
  
  // LLM layers (similar to Qwen3)
  std::unique_ptr<Qwen3Layers> qwen_layers_;
  
  // Vision operation layers (wrapping kernel calls as layer->forward())
  mutable VisionVLLayers vision_vl_layers_;
  
  // Intermediate buffers for vision encoder
  mutable std::vector<tensor::Tensor> vision_buffers_;
  
  // Pre-allocated workspace for vision transformer blocks
  mutable std::unique_ptr<VisionWorkspace> vision_workspace_;
  
  // Deepstack feature storage
  mutable std::vector<tensor::Tensor> deepstack_features_;
  
  // Visual position markers for deepstack processing
  // These mark the range [visual_pos_start_, visual_pos_end_) in the sequence
  // that corresponds to visual tokens
  mutable int visual_pos_start_ = -1;
  mutable int visual_pos_end_ = -1;
  
  // M-RoPE 3D position storage for multimodal inputs
  // For each token in the sequence, we store (pos_t, pos_h, pos_w)
  // Visual tokens: (0, row_idx, col_idx)
  // Text tokens: (seq_pos, seq_pos, seq_pos)
  mutable std::vector<int32_t> mrope_pos_t_;  // Temporal positions
  mutable std::vector<int32_t> mrope_pos_h_;  // Height positions
  mutable std::vector<int32_t> mrope_pos_w_;  // Width positions
  mutable int mrope_max_text_pos_ = 0;        // Track max text position for decode
  
  // GPU-resident M-RoPE position arrays for batched kernel
  // OPTIMIZED: Use single contiguous allocation for all 3 arrays
  mutable int32_t* mrope_pos_gpu_ = nullptr;  // Contiguous: [t, h, w] interleaved
  mutable int32_t* mrope_pos_t_gpu_ = nullptr;  // Points into mrope_pos_gpu_
  mutable int32_t* mrope_pos_h_gpu_ = nullptr;
  mutable int32_t* mrope_pos_w_gpu_ = nullptr;
  mutable size_t mrope_pos_gpu_capacity_ = 0;  // Allocated capacity per array
  
  // Pinned host memory for M-RoPE positions (async transfer)
  mutable int32_t* mrope_pos_pinned_ = nullptr;
  mutable size_t mrope_pos_pinned_capacity_ = 0;
  
  // Prefill sequence length (for decode position calculation)
  mutable int prefill_seq_len_ = 0;
  
  // Cached image data for repeated use
  mutable ImageData cached_image_data_;
  
  // mmap for model file
  void* vl_model_data_ = nullptr;
  size_t vl_model_file_size_ = 0;
  int vl_model_fd_ = -1;
};

/**
 * @brief Image preprocessing utilities
 */
namespace image_utils {

/**
 * @brief Load image from file path
 * @param path Image file path
 * @param width Output image width
 * @param height Output image height
 * @param channels Output channels (should be 3 for RGB)
 * @return Raw pixel data (HWC format, uint8)
 */
std::vector<uint8_t> load_image(const std::string& path, 
                                 int& width, int& height, int& channels);

/**
 * @brief Smart resize for Qwen3-VL
 * Resize image to match patch_size * merge_size requirements
 * 
 * @param pixels Input pixels
 * @param src_width Source width
 * @param src_height Source height
 * @param min_pixels Minimum total pixels (default: 256*28*28 = 200704)
 * @param max_pixels Maximum total pixels (default: 1280*28*28 = 1003520)
 * @param factor Must be divisible by (patch_size * merge_size = 16*2 = 32)
 * @return Resized pixels and new dimensions
 */
std::tuple<std::vector<uint8_t>, int, int> smart_resize(
    const std::vector<uint8_t>& pixels,
    int src_width, int src_height,
    int min_pixels = 200704,
    int max_pixels = 1003520,
    int factor = 32);

/**
 * @brief Normalize and convert to tensor
 * Applies ImageNet normalization: (x/255 - mean) / std
 * mean = [0.485, 0.456, 0.406]
 * std = [0.229, 0.224, 0.225]
 * 
 * @param pixels RGB pixels (HWC, uint8)
 * @param width Image width
 * @param height Image height
 * @return FP16 tensor [3, height, width]
 */
tensor::Tensor normalize_to_tensor(const std::vector<uint8_t>& pixels,
                                    int width, int height);

/**
 * @brief Convert image to patches for vision encoder
 * 
 * @param image_tensor Normalized image [3, H, W]
 * @param patch_size Patch size (16)
 * @param temporal_patch_size Temporal patch size (2, padded to 1 frame)
 * @return Patch tensor [num_patches, patch_dim]
 */
tensor::Tensor image_to_patches(const tensor::Tensor& image_tensor,
                                 int patch_size = 16,
                                 int temporal_patch_size = 2);

} // namespace image_utils

}  // namespace model

#endif  // KUIPER_INCLUDE_MODEL_QWEN3_VL_H_
