#ifndef KUIPER_OP_KERNELS_CPU_IMAGE_PREPROCESS_KERNEL_H_
#define KUIPER_OP_KERNELS_CPU_IMAGE_PREPROCESS_KERNEL_H_

#include <cstdint>
#include <vector>
#include <string>
#include <tuple>

namespace kernel {

/**
 * @brief Load image from file path (CPU)
 */
std::vector<uint8_t> load_image_cpu(const std::string& path,
                                     int& width, int& height, int& channels);

/**
 * @brief Smart resize for Qwen3-VL (CPU)
 */
std::tuple<std::vector<uint8_t>, int, int> smart_resize_cpu(
    const std::vector<uint8_t>& pixels,
    int src_width, int src_height,
    int min_pixels, int max_pixels, int factor);

/**
 * @brief Compute smart resize target dimensions only (no actual resize).
 * Mirrors the integer math in smart_resize_cpu without touching pixel data.
 * Used by the GPU-fused resize path (9.4 optimization).
 */
void smart_resize_calc_dims(int src_width, int src_height,
                            int min_pixels, int max_pixels, int factor,
                            int& new_width, int& new_height);

/**
 * @brief Float32 to float16 conversion (CPU)
 */
uint16_t float_to_half_cpu(float value);

/**
 * @brief Compute vision rotary embeddings (CPU)
 * @param cos_data Output cosine embeddings [num_tokens * head_dim]
 * @param sin_data Output sine embeddings [num_tokens * head_dim]
 * @param grid_h Grid height
 * @param grid_w Grid width
 * @param grid_t Grid temporal (usually 1)
 * @param num_heads Number of attention heads
 * @param hidden_size Hidden size
 * @param spatial_merge_size Spatial merge size
 */
void compute_vision_rotary_emb_cpu(
    std::vector<uint16_t>& cos_data,
    std::vector<uint16_t>& sin_data,
    int grid_h, int grid_w, int grid_t,
    int num_heads, int hidden_size, int spatial_merge_size);

/**
 * @brief Generate M-RoPE 3D positions (CPU)
 * @param mrope_pos_t Output temporal positions
 * @param mrope_pos_h Output height positions
 * @param mrope_pos_w Output width positions
 * @param tokens Token IDs (used for sequence length)
 * @param image_token_pos Position of image token
 * @param num_vision_tokens Number of vision tokens
 * @param grid_h Grid height (merged)
 * @param grid_w Grid width (merged)
 * @return max_text_pos
 */
int generate_mrope_positions_cpu(
    std::vector<int32_t>& mrope_pos_t,
    std::vector<int32_t>& mrope_pos_h,
    std::vector<int32_t>& mrope_pos_w,
    const std::vector<int>& tokens,
    int image_token_pos, int num_vision_tokens,
    int grid_h, int grid_w);

}  // namespace kernel

#endif  // KUIPER_OP_KERNELS_CPU_IMAGE_PREPROCESS_KERNEL_H_
