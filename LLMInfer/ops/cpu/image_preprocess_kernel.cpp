#include "image_preprocess_kernel.h"
#include <cmath>
#include <cstring>
#include <algorithm>
#include <glog/logging.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb/stb_image_resize2.h"

namespace kernel {

std::vector<uint8_t> load_image_cpu(const std::string& path,
                                     int& width, int& height, int& channels) {
  unsigned char* data = stbi_load(path.c_str(), &width, &height, &channels, 3);
  if (!data) {
    LOG(ERROR) << "Failed to load image: " << path;
    return {};
  }
  
  channels = 3;
  std::vector<uint8_t> pixels(data, data + width * height * channels);
  stbi_image_free(data);
  
  LOG(INFO) << "Loaded image: " << path << " (" << width << "x" << height << "x" << channels << ")";
  return pixels;
}

std::tuple<std::vector<uint8_t>, int, int> smart_resize_cpu(
    const std::vector<uint8_t>& pixels,
    int src_width, int src_height,
    int min_pixels, int max_pixels, int factor) {
  
  int h_bar = static_cast<int>(std::round(static_cast<float>(src_height) / factor)) * factor;
  int w_bar = static_cast<int>(std::round(static_cast<float>(src_width) / factor)) * factor;
  
  h_bar = std::max(h_bar, factor);
  w_bar = std::max(w_bar, factor);
  
  if (h_bar * w_bar > max_pixels) {
    float beta = std::sqrt(static_cast<float>(src_height * src_width) / max_pixels);
    h_bar = std::max(factor, static_cast<int>(std::floor(src_height / beta / factor)) * factor);
    w_bar = std::max(factor, static_cast<int>(std::floor(src_width / beta / factor)) * factor);
  } else if (h_bar * w_bar < min_pixels) {
    float beta = std::sqrt(static_cast<float>(min_pixels) / (src_height * src_width));
    h_bar = static_cast<int>(std::ceil(src_height * beta / factor)) * factor;
    w_bar = static_cast<int>(std::ceil(src_width * beta / factor)) * factor;
  }
  
  LOG(INFO) << "Smart resize: " << src_width << "x" << src_height 
            << " -> " << w_bar << "x" << h_bar;
  
  std::vector<uint8_t> resized(w_bar * h_bar * 3);
  stbir_resize_uint8_linear(
      pixels.data(), src_width, src_height, src_width * 3,
      resized.data(), w_bar, h_bar, w_bar * 3,
      STBIR_RGB);
  
  return {resized, w_bar, h_bar};
}

void smart_resize_calc_dims(int src_width, int src_height,
                            int min_pixels, int max_pixels, int factor,
                            int& new_width, int& new_height) {
  int h_bar = static_cast<int>(std::round(static_cast<float>(src_height) / factor)) * factor;
  int w_bar = static_cast<int>(std::round(static_cast<float>(src_width)  / factor)) * factor;
  h_bar = std::max(h_bar, factor);
  w_bar = std::max(w_bar, factor);
  if (h_bar * w_bar > max_pixels) {
    float beta = std::sqrt(static_cast<float>(src_height * src_width) / max_pixels);
    h_bar = std::max(factor, static_cast<int>(std::floor(src_height / beta / factor)) * factor);
    w_bar = std::max(factor, static_cast<int>(std::floor(src_width  / beta / factor)) * factor);
  } else if (h_bar * w_bar < min_pixels) {
    float beta = std::sqrt(static_cast<float>(min_pixels) / (src_height * src_width));
    h_bar = static_cast<int>(std::ceil(src_height * beta / factor)) * factor;
    w_bar = static_cast<int>(std::ceil(src_width  * beta / factor)) * factor;
  }
  new_width = w_bar;
  new_height = h_bar;
}

uint16_t float_to_half_cpu(float value) {
  if (value == 0.0f) return 0;
  if (std::isnan(value)) return 0x7e00;
  if (std::isinf(value)) return (value > 0) ? 0x7c00 : 0xfc00;
  
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  
  uint32_t sign = (bits >> 31) & 0x1;
  int32_t exp = ((bits >> 23) & 0xff) - 127;
  uint32_t frac = bits & 0x7fffff;
  
  uint16_t h_sign = sign << 15;
  uint16_t h_exp;
  uint16_t h_frac;
  
  if (exp < -24) {
    return h_sign;
  } else if (exp < -14) {
    h_exp = 0;
    h_frac = (frac | 0x800000) >> (14 - exp);
  } else if (exp > 15) {
    return h_sign | 0x7c00;
  } else {
    h_exp = (exp + 15) << 10;
    h_frac = frac >> 13;
  }
  
  return h_sign | h_exp | h_frac;
}

void compute_vision_rotary_emb_cpu(
    std::vector<uint16_t>& cos_data,
    std::vector<uint16_t>& sin_data,
    int grid_h, int grid_w, int grid_t,
    int num_heads, int hidden_size, int spatial_merge_size) {
  
  int head_dim = hidden_size / num_heads;
  int half_head_dim = head_dim / 2;
  int quarter_head_dim = head_dim / 4;
  
  int merge_size = spatial_merge_size;
  int merged_h = grid_h / merge_size;
  int merged_w = grid_w / merge_size;
  int num_tokens = grid_t * grid_h * grid_w;
  int max_hw = std::max(grid_h, grid_w);
  
  float theta = 10000.0f;
  
  // Compute inverse frequencies
  std::vector<float> inv_freq(quarter_head_dim);
  for (int i = 0; i < quarter_head_dim; ++i) {
    inv_freq[i] = 1.0f / std::pow(theta, static_cast<float>(2 * i) / half_head_dim);
  }
  
  // Compute frequency table
  std::vector<float> freq_table(max_hw * quarter_head_dim);
  for (int seq = 0; seq < max_hw; ++seq) {
    for (int i = 0; i < quarter_head_dim; ++i) {
      freq_table[seq * quarter_head_dim + i] = seq * inv_freq[i];
    }
  }
  
  // Compute position IDs in spatial merge order
  std::vector<int32_t> pos_h(num_tokens);
  std::vector<int32_t> pos_w(num_tokens);
  int token_idx = 0;
  for (int t = 0; t < grid_t; ++t) {
    for (int block_h = 0; block_h < merged_h; ++block_h) {
      for (int block_w = 0; block_w < merged_w; ++block_w) {
        for (int local_h = 0; local_h < merge_size; ++local_h) {
          for (int local_w = 0; local_w < merge_size; ++local_w) {
            pos_h[token_idx] = block_h * merge_size + local_h;
            pos_w[token_idx] = block_w * merge_size + local_w;
            ++token_idx;
          }
        }
      }
    }
  }
  
  // Compute cos/sin with float_to_half
  cos_data.resize(num_tokens * head_dim);
  sin_data.resize(num_tokens * head_dim);
  
  for (int i = 0; i < num_tokens; ++i) {
    int h_pos = pos_h[i];
    int w_pos = pos_w[i];
    
    for (int j = 0; j < quarter_head_dim; ++j) {
      float h_freq = freq_table[h_pos * quarter_head_dim + j];
      float w_freq = freq_table[w_pos * quarter_head_dim + j];
      
      cos_data[i * head_dim + j] = float_to_half_cpu(std::cos(h_freq));
      sin_data[i * head_dim + j] = float_to_half_cpu(std::sin(h_freq));
      
      cos_data[i * head_dim + quarter_head_dim + j] = float_to_half_cpu(std::cos(w_freq));
      sin_data[i * head_dim + quarter_head_dim + j] = float_to_half_cpu(std::sin(w_freq));
      
      cos_data[i * head_dim + half_head_dim + j] = float_to_half_cpu(std::cos(h_freq));
      sin_data[i * head_dim + half_head_dim + j] = float_to_half_cpu(std::sin(h_freq));
      
      cos_data[i * head_dim + half_head_dim + quarter_head_dim + j] = float_to_half_cpu(std::cos(w_freq));
      sin_data[i * head_dim + half_head_dim + quarter_head_dim + j] = float_to_half_cpu(std::sin(w_freq));
    }
  }
}

int generate_mrope_positions_cpu(
    std::vector<int32_t>& mrope_pos_t,
    std::vector<int32_t>& mrope_pos_h,
    std::vector<int32_t>& mrope_pos_w,
    const std::vector<int>& tokens,
    int image_token_pos, int num_vision_tokens,
    int grid_h, int grid_w) {
  
  int new_seq_len = static_cast<int>(tokens.size()) - 1 + num_vision_tokens;
  mrope_pos_t.resize(new_seq_len);
  mrope_pos_h.resize(new_seq_len);
  mrope_pos_w.resize(new_seq_len);
  
  // Text tokens before image
  int text_pos = 0;
  for (int i = 0; i < image_token_pos; ++i) {
    mrope_pos_t[i] = text_pos;
    mrope_pos_h[i] = text_pos;
    mrope_pos_w[i] = text_pos;
    text_pos++;
  }
  
  // Visual tokens - use 2D spatial positions
  int visual_base_t = text_pos;
  for (int v = 0; v < num_vision_tokens; ++v) {
    int row = v / grid_w;
    int col = v % grid_w;
    mrope_pos_t[image_token_pos + v] = visual_base_t;
    mrope_pos_h[image_token_pos + v] = visual_base_t + row;
    mrope_pos_w[image_token_pos + v] = visual_base_t + col;
  }
  
  // Text tokens after image
  int max_visual_extent = std::max(grid_h, grid_w);
  text_pos = visual_base_t + max_visual_extent;
  
  int after_image_start = image_token_pos + num_vision_tokens;
  for (int i = after_image_start; i < new_seq_len; ++i) {
    mrope_pos_t[i] = text_pos;
    mrope_pos_h[i] = text_pos;
    mrope_pos_w[i] = text_pos;
    text_pos++;
  }
  
  int mrope_max_text_pos = text_pos - 1;
  
  LOG(INFO) << "M-RoPE positions generated: "
            << "visual_base_t=" << visual_base_t
            << ", merged_grid=" << grid_h << "x" << grid_w
            << ", max_text_pos=" << mrope_max_text_pos;
  
  return mrope_max_text_pos;
}

}  // namespace kernel
