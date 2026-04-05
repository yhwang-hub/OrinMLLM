/**
 * @file main_qwen3_5.cpp
 * @brief Qwen3.5-9B Hybrid Vision-Language Model Demo
 *
 * Usage:
 *   ./demo/qwen3_5_infer model.bin tokenizer.json --image demo.jpeg
 *       --prompt "Describe this image." --stream --max-tokens 256
 */

#include <base/base.h>
#include <base/tick.h>
#include <glog/logging.h>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <string>
#include <getopt.h>
#include "inference_common.h"

#ifdef QWEN3_VL_SUPPORT
#include "model/qwen3_5.h"
#endif

struct Q35InferenceConfig {
  std::string model_path;
  std::string token_path;
  std::string image_path;
  std::string prompt = "Describe this image.";
  int max_tokens = 256;
  int max_pixels = 1003520;
  bool stream_output = false;
  bool use_cuda_graph = false;
};

struct PerformanceStats {
  double image_preprocess_time_ms = 0.0;
  double vit_encode_time_ms = 0.0;
  double vit_embedding_time_ms = 0.0;
  double vit_total_time_ms = 0.0;
  double vit_prefill_transition_time_ms = 0.0;
  double prefill_time_ms = 0.0;
  int num_prefill_tokens = 0;
  double decode_time_ms = 0.0;
  int num_decode_tokens = 0;

  void print() const {
    std::cout << "\n=== Performance Statistics ===\n";
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  Image Preprocessing:\n";
    std::cout << "    Time: " << image_preprocess_time_ms << " ms\n";
    std::cout << "  ViT (Vision Encoder):\n";
    std::cout << "    Total Time: " << vit_total_time_ms << " ms\n";
    std::cout << "  ViT->Prefill Transition:\n";
    std::cout << "    Time: " << vit_prefill_transition_time_ms << " ms\n";
    std::cout << "  Prefill:\n";
    std::cout << "    Tokens: " << num_prefill_tokens << "\n";
    std::cout << "    Time: " << prefill_time_ms << " ms\n";
    std::cout << "    Throughput: " << (num_prefill_tokens > 0 ? (num_prefill_tokens * 1000.0 / prefill_time_ms) : 0) << " tokens/s\n";
    std::cout << "  Decode:\n";
    std::cout << "    Tokens: " << num_decode_tokens << "\n";
    std::cout << "    Time: " << decode_time_ms << " ms\n";
    std::cout << "    Throughput: " << (num_decode_tokens > 0 ? (num_decode_tokens * 1000.0 / decode_time_ms) : 0) << " tokens/s\n";
    std::cout << "    Latency: " << (num_decode_tokens > 0 ? (decode_time_ms / num_decode_tokens) : 0) << " ms/token\n";
    std::cout << "  Total:\n";
    std::cout << "    Time: " << (image_preprocess_time_ms + vit_total_time_ms + vit_prefill_transition_time_ms + prefill_time_ms + decode_time_ms) << " ms\n";
    std::cout << "==============================\n";
  }
};

void print_usage(const char* prog) {
  std::cout << "Usage: " << prog << " <model.bin> <tokenizer.json> [options]\n"
            << "\nOptions:\n"
            << "  --image <path>       Input image path (required)\n"
            << "  --prompt <text>      User prompt (default: 'Describe this image.')\n"
            << "  --max-tokens <n>     Max tokens to generate (default: 256)\n"
            << "  --max-pixels <n>     Max image pixels (default: 1003520)\n"
            << "  --stream             Enable streaming output\n"
            << "  --cuda-graph         Enable CUDA Graph for decode (experimental)\n"
            << "  -h, --help           Show help\n";
}

Q35InferenceConfig parse_args(int argc, char* argv[]) {
  Q35InferenceConfig config;
  
  static struct option opts[] = {
    {"image", required_argument, 0, 'i'},
    {"prompt", required_argument, 0, 'p'},
    {"max-tokens", required_argument, 0, 'm'},
    {"max-pixels", required_argument, 0, 'x'},
    {"stream", no_argument, 0, 's'},
    {"cuda-graph", no_argument, 0, 'g'},
    {"help", no_argument, 0, 'h'},
    {0, 0, 0, 0}
  };
  
  int opt, idx = 0;
  while ((opt = getopt_long(argc, argv, "i:p:m:x:sgh", opts, &idx)) != -1) {
    switch (opt) {
      case 'i': config.image_path = optarg; break;
      case 'p': config.prompt = optarg; break;
      case 'm': config.max_tokens = std::stoi(optarg); break;
      case 'x': config.max_pixels = std::stoi(optarg); break;
      case 's': config.stream_output = true; break;
      case 'g': config.use_cuda_graph = true; break;
      case 'h': print_usage(argv[0]); exit(0);
      default: print_usage(argv[0]); exit(1);
    }
  }
  
  if (optind < argc) config.model_path = argv[optind++];
  if (optind < argc) config.token_path = argv[optind++];
  
  return config;
}

#ifdef QWEN3_VL_SUPPORT
int run_inference(const Q35InferenceConfig& config) {
  bool has_image = !config.image_path.empty();
  LOG(INFO) << "=== Qwen3.5-9B Vision-Language Model Inference ===";
  LOG(INFO) << "Model: " << config.model_path;
  if (has_image) LOG(INFO) << "Image: " << config.image_path;
  LOG(INFO) << "Prompt: " << config.prompt;
  
  // Create model
  model::Qwen35Model model(base::TokenizerType::kEncodeBpe,
                           config.token_path, config.model_path);
  
  LOG(INFO) << "Initializing model...";
  auto status = model.init(base::DeviceType::kDeviceCUDA);
  if (!status) {
    LOG(ERROR) << "Model init failed: " << status.get_err_code();
    return 1;
  }
  LOG(INFO) << "Model initialized";

  // Enable CUDA Graph if requested
  if (config.use_cuda_graph) {
    model.enable_cuda_graph(true);
    LOG(INFO) << "CUDA Graph optimization enabled";
  }
  
  // Build prompt using Qwen3.5 chat template
  std::string full_prompt;
  if (has_image) {
    full_prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                  "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
                  + config.prompt + "<|im_end|>\n"
                  "<|im_start|>assistant\n";
  } else {
    full_prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                  "<|im_start|>user\n"
                  + config.prompt + "<|im_end|>\n"
                  "<|im_start|>assistant\n";
  }
  
  auto tokens = model.encode(full_prompt);
  LOG(INFO) << "Prompt tokens: " << tokens.size();
  
  PerformanceStats perf_stats;
  int prefill_len = 0;
  tensor::Tensor embeddings;
  int eos_id = model.get_vl_config().special_tokens.eos_token_id;
  
  if (has_image) {
    // Stage 1: Image Preprocessing
    LOG(INFO) << "\n>>> Stage 1: Image Preprocessing <<<";
    auto t0 = std::chrono::high_resolution_clock::now();
    auto image_data = model.preprocess_image(config.image_path, config.max_pixels);
    cudaDeviceSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();
    perf_stats.image_preprocess_time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    LOG(INFO) << "Image: " << image_data.num_patches << " patches -> " 
              << image_data.num_vision_tokens << " vision tokens";
    LOG(INFO) << "Image preprocessing time: " << perf_stats.image_preprocess_time_ms << " ms";
    
    // Stage 2: ViT + Embedding
    LOG(INFO) << "\n>>> Stage 2: Vision Encoder (ViT) <<<";
    auto t2 = std::chrono::high_resolution_clock::now();
    embeddings = model.prepare_multimodal_embeddings(tokens, &image_data);
    cudaDeviceSynchronize();
    auto t3 = std::chrono::high_resolution_clock::now();
    perf_stats.vit_total_time_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
    prefill_len = static_cast<int>(tokens.size()) - 1 + image_data.num_vision_tokens;
    LOG(INFO) << "ViT + Embedding: " << prefill_len << " tokens in " << perf_stats.vit_total_time_ms << " ms";
    
    // Stage 2.5: Transition
    auto t4 = std::chrono::high_resolution_clock::now();
    cudaDeviceSynchronize();
    auto t5 = std::chrono::high_resolution_clock::now();
    perf_stats.vit_prefill_transition_time_ms = std::chrono::duration<double, std::milli>(t5 - t4).count();
  } else {
    LOG(INFO) << "Text-only mode: embedding tokens...";
    auto t0 = std::chrono::high_resolution_clock::now();
    embeddings = model.prepare_multimodal_embeddings(tokens, nullptr);
    cudaDeviceSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();
    perf_stats.vit_total_time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    prefill_len = static_cast<int>(tokens.size());
  }
  perf_stats.num_prefill_tokens = prefill_len;
  LOG(INFO) << "Prefill length: " << prefill_len;
  
  // Stage 3: Prefill
  LOG(INFO) << "\n>>> Stage 3: Prefill <<<";
  auto t_prefill_start = std::chrono::high_resolution_clock::now();
  status = model.prefill(embeddings, prefill_len, 0);
  if (!status) {
    LOG(ERROR) << "Prefill failed";
    return 1;
  }
  int next_token = model.sample_first_token();
  cudaDeviceSynchronize();
  auto t_prefill_end = std::chrono::high_resolution_clock::now();
  perf_stats.prefill_time_ms = std::chrono::duration<double, std::milli>(t_prefill_end - t_prefill_start).count();
  LOG(INFO) << "Prefill complete in " << perf_stats.prefill_time_ms << " ms";
  LOG(INFO) << "First token: " << next_token << " = '" << model.decode(next_token) << "'";
  
  if (next_token == eos_id) {
    LOG(INFO) << "EOS token, no generation needed";
    perf_stats.print();
    return 0;
  }
  
  // Stage 4: Decode
  LOG(INFO) << "\n>>> Stage 4: Decode (Auto-regressive Generation) <<<";
  std::string response;
  int decode_tokens = 0;
  
  if (config.stream_output) {
    std::cout << "\n=== Response (Streaming) ===\n" << std::flush;
    std::string first_str = model.decode(next_token);
    std::cout << first_str << std::flush;
    response += first_str;
  }
  
  auto t_decode_start = std::chrono::high_resolution_clock::now();
  
  for (int step = 0; step < config.max_tokens; ++step) {
    if (next_token == eos_id) break;
    
    if (!config.stream_output || step > 0) {
      std::string piece = model.decode(next_token);
      response += piece;
      if (config.stream_output) std::cout << piece << std::flush;
    }
    
    model.embedding_to_decode_input(next_token);
    int pos = prefill_len + step;
    status = model.decode_step_optimized(pos, next_token);
    if (!status) {
      LOG(ERROR) << "Decode step failed at pos=" << pos;
      break;
    }
    decode_tokens++;
  }
  
  cudaDeviceSynchronize();
  auto t_decode_end = std::chrono::high_resolution_clock::now();
  perf_stats.decode_time_ms = std::chrono::duration<double, std::milli>(t_decode_end - t_decode_start).count();
  perf_stats.num_decode_tokens = decode_tokens;
  
  if (config.stream_output) std::cout << std::endl;
  
  std::cout << "\n=== Response ===" << std::endl;
  std::cout << response << std::endl;
  std::cout << "================\n" << std::endl;
  
  perf_stats.print();
  
  return 0;
}
#endif

int main(int argc, char* argv[]) {
  google::InitGoogleLogging(argv[0]);
  FLAGS_logtostderr = true;
  FLAGS_minloglevel = 0;
  
  auto config = parse_args(argc, argv);
  
  if (config.model_path.empty() || config.token_path.empty()) {
    print_usage(argv[0]);
    return 1;
  }
  
  // Allow text-only mode (no image)
  bool has_image = !config.image_path.empty();
  
  // Validate files exist
  std::vector<std::string> check_files = {config.model_path, config.token_path};
  if (has_image) check_files.push_back(config.image_path);
  for (auto& path : check_files) {
    std::ifstream f(path);
    if (!f.good()) {
      LOG(ERROR) << "File not found: " << path;
      return 1;
    }
  }
  
#ifdef QWEN3_VL_SUPPORT
  return run_inference(config);
#else
  LOG(ERROR) << "QWEN3_VL_SUPPORT not enabled. Rebuild with -DQWEN3_VL_SUPPORT=ON";
  return 1;
#endif
}
