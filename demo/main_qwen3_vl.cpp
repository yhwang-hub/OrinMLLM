/**
 * @file main_qwen3_vl.cpp
 * @brief Qwen3-VL Vision-Language Model Demo
 * 
 * 运行示例:
 *   ./demo/qwen3_vl_infer model.bin tokenizer.json --image demo.jpeg
 * 
 * 参数说明:
 *   model.bin       - Qwen3-VL-8B FP16模型文件
 *   tokenizer.json  - Qwen3 tokenizer文件
 *   --image         - 输入图片路径
 *   --prompt        - 用户提示词 (默认: "Describe this image.")
 *   --max-tokens    - 最大生成token数 (默认: 256)
 *   --stream        - 流式输出
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
#include "model/qwen3_vl.h"
#endif

// Performance statistics structure
struct PerformanceStats {
  // Image preprocessing
  double image_preprocess_time_ms = 0.0;
  
  // ViT (Vision Encoder)
  double vit_encode_time_ms = 0.0;       // encode_image (actual ViT forward)
  double vit_embedding_time_ms = 0.0;    // multimodal embedding assembly
  double vit_total_time_ms = 0.0;        // total ViT stage time
  
  // ViT -> Prefill transition
  double vit_prefill_transition_time_ms = 0.0;
  
  // Prefill
  double prefill_time_ms = 0.0;
  int num_prefill_tokens = 0;
  
  // Decode
  double decode_time_ms = 0.0;
  int num_decode_tokens = 0;
  
  void print() const {
    std::cout << "\n=== Performance Statistics ===\n";
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  Image Preprocessing:\n";
    std::cout << "    Time: " << image_preprocess_time_ms << " ms\n";
    std::cout << "  ViT (Vision Encoder):\n";
    std::cout << "    Encode Time: " << vit_encode_time_ms << " ms\n";
    std::cout << "    Embedding Assembly Time: " << vit_embedding_time_ms << " ms\n";
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

struct VLInferenceConfig {
  std::string model_path;
  std::string token_path;
  std::string image_path;
  std::string prompt = "Describe this image.";
  int max_tokens = 256;
  int max_pixels = 1003520;  // 14*14*4*1280, lower = faster ViT
  bool stream_output = false;
  bool verbose = false;
  bool use_cuda_graph = false;
};

void print_usage(const char* program_name) {
  std::cout << "Usage: " << program_name << " <model.bin> <tokenizer.json> [options]\n"
            << "\nOptions:\n"
            << "  --image <path>       Input image path (required)\n"
            << "  --prompt <text>      User prompt (default: 'Describe this image.')\n"
            << "  --max-tokens <n>     Maximum tokens to generate (default: 256)\n"
            << "  --max-pixels <n>     Maximum image pixels (default: 1003520)\n"
            << "                       Lower = faster ViT. Suggested: 1003520, 500000, 400000\n"
            << "  --stream             Enable streaming output\n"
            << "  --cuda-graph         Enable CUDA Graph for faster decode\n"
            << "  --verbose            Enable verbose logging\n"
            << "  -h, --help           Show this help message\n"
            << "\nExample:\n"
            << "  " << program_name << " /path/to/Qwen3-VL-8B-fp16.bin /path/to/tokenizer.json \\\n"
            << "      --image /path/to/demo.jpeg --prompt 'What is in this image?' --cuda-graph\n";
}

VLInferenceConfig parse_args(int argc, char* argv[]) {
  VLInferenceConfig config;
  
  static struct option long_options[] = {
    {"image", required_argument, 0, 'i'},
    {"prompt", required_argument, 0, 'p'},
    {"max-tokens", required_argument, 0, 'm'},
    {"max-pixels", required_argument, 0, 'x'},
    {"stream", no_argument, 0, 's'},
    {"cuda-graph", no_argument, 0, 'g'},
    {"verbose", no_argument, 0, 'v'},
    {"help", no_argument, 0, 'h'},
    {0, 0, 0, 0}
  };
  
  int opt;
  int option_index = 0;
  
  while ((opt = getopt_long(argc, argv, "i:p:m:x:sgvh", long_options, &option_index)) != -1) {
    switch (opt) {
      case 'i':
        config.image_path = optarg;
        break;
      case 'p':
        config.prompt = optarg;
        break;
      case 'm':
        config.max_tokens = std::stoi(optarg);
        break;
      case 'x':
        config.max_pixels = std::stoi(optarg);
        break;
      case 's':
        config.stream_output = true;
        break;
      case 'g':
        config.use_cuda_graph = true;
        break;
      case 'v':
        config.verbose = true;
        break;
      case 'h':
        print_usage(argv[0]);
        exit(0);
      default:
        print_usage(argv[0]);
        exit(1);
    }
  }
  
  // Parse positional arguments
  if (optind < argc) {
    config.model_path = argv[optind++];
  }
  if (optind < argc) {
    config.token_path = argv[optind++];
  }
  
  return config;
}

bool validate_config(const VLInferenceConfig& config) {
  bool valid = true;
  
  if (config.model_path.empty()) {
    LOG(ERROR) << "Model path is required";
    valid = false;
  } else {
    std::ifstream f(config.model_path);
    if (!f.good()) {
      LOG(ERROR) << "Model file not found: " << config.model_path;
      valid = false;
    }
  }
  
  if (config.token_path.empty()) {
    LOG(ERROR) << "Tokenizer path is required";
    valid = false;
  } else {
    std::ifstream f(config.token_path);
    if (!f.good()) {
      LOG(ERROR) << "Tokenizer file not found: " << config.token_path;
      valid = false;
    }
  }
  
  if (config.image_path.empty()) {
    LOG(ERROR) << "Image path is required (use --image <path>)";
    valid = false;
  } else {
    std::ifstream f(config.image_path);
    if (!f.good()) {
      LOG(ERROR) << "Image file not found: " << config.image_path;
      valid = false;
    }
  }
  
  return valid;
}

#ifdef QWEN3_VL_SUPPORT
int run_inference(const VLInferenceConfig& config) {
  LOG(INFO) << "=== Qwen3-VL Vision-Language Model Inference ===";
  LOG(INFO) << "Model: " << config.model_path;
  LOG(INFO) << "Image: " << config.image_path;
  LOG(INFO) << "Prompt: " << config.prompt;
  LOG(INFO) << "Max tokens: " << config.max_tokens;
  LOG(INFO) << "Stream output: " << (config.stream_output ? "enabled" : "disabled");
  LOG(INFO) << "CUDA Graph: " << (config.use_cuda_graph ? "enabled" : "disabled");
  
  // Create model
  model::Qwen3VLModel model(base::TokenizerType::kEncodeBpe,
                            config.token_path,
                            config.model_path);
  
  // Initialize
  LOG(INFO) << "Initializing model...";
  inference::Timer init_timer;
  init_timer.start();
  
  auto status = model.init(base::DeviceType::kDeviceCUDA);
  if (!status) {
    LOG(ERROR) << "Model initialization failed: " << status.get_err_code();
    return 1;
  }
  
  // Enable CUDA Graph if requested
  if (config.use_cuda_graph) {
    model.enable_cuda_graph(true);
    LOG(INFO) << "CUDA Graph optimization enabled";
  }
  
  double init_time = init_timer.elapsed_ms();
  LOG(INFO) << "Model initialized in " << init_time << " ms";
  
  // Print model config
  const auto& vl_config = model.get_vl_config();
  LOG(INFO) << "\nModel Configuration:";
  LOG(INFO) << "  Vision: " << vl_config.vision.depth << " layers, "
            << "hidden=" << vl_config.vision.hidden_size;
  LOG(INFO) << "  LLM: " << vl_config.text.num_hidden_layers << " layers, "
            << "hidden=" << vl_config.text.hidden_size;
  LOG(INFO) << "  Vocab size: " << vl_config.text.vocab_size;
  
  // Build prompt with chat template
  std::string full_prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                            "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
                            + config.prompt + "<|im_end|>\n"
                            "<|im_start|>assistant\n";
  
  // Tokenize prompt
  auto tokens = model.encode(full_prompt);
  LOG(INFO) << "Prompt tokens: " << tokens.size();
  
  // Performance statistics
  PerformanceStats perf_stats;
  
  // ========================================
  // Stage 1: Image Preprocessing
  // ========================================
  LOG(INFO) << "\n>>> Stage 1: Image Preprocessing <<<";
  LOG(INFO) << "Using max_pixels=" << config.max_pixels;
  
  inference::Timer preprocess_timer;
  preprocess_timer.start();
  
  auto image_data = model.preprocess_image(config.image_path, config.max_pixels);
  
  cudaDeviceSynchronize();
  perf_stats.image_preprocess_time_ms = preprocess_timer.elapsed_ms();
  
  LOG(INFO) << "Image preprocessed: " << image_data.num_patches << " patches -> " 
            << image_data.num_vision_tokens << " vision tokens";
  LOG(INFO) << "Image preprocessing time: " << perf_stats.image_preprocess_time_ms << " ms";
  
  // ========================================
  // Stage 2: Vision Encoder (ViT) + Multimodal Embedding
  // ========================================
  LOG(INFO) << "\n>>> Stage 2: Vision Encoder (ViT) <<<";
  inference::Timer vit_timer;
  vit_timer.start();
  
  // prepare_multimodal_embeddings internally calls encode_image (ViT)
  auto embeddings = model.prepare_multimodal_embeddings(tokens, &image_data);
  
  cudaDeviceSynchronize();
  perf_stats.vit_total_time_ms = vit_timer.elapsed_ms();
  
  // Note: For more detailed breakdown, the model internally tracks:
  // - encode_image time (ViT forward)
  // - embedding assembly time
  // These are logged by the model, here we capture the total time
  
  // Calculate prefill sequence length
  int prefill_seq_len = static_cast<int>(tokens.size()) - 1 + image_data.num_vision_tokens;
  perf_stats.num_prefill_tokens = prefill_seq_len;
  LOG(INFO) << "ViT + Embedding complete: " << prefill_seq_len << " tokens in " 
            << perf_stats.vit_total_time_ms << " ms";
  
  // ========================================
  // Stage 2.5: ViT -> Prefill Transition
  // ========================================
  LOG(INFO) << "\n>>> Stage 2.5: ViT -> Prefill Transition <<<";
  inference::Timer transition_timer;
  transition_timer.start();
  
  // This includes any data preparation between ViT output and prefill input
  // Most of the work is already done in prepare_multimodal_embeddings,
  // but we measure any remaining setup time here
  cudaDeviceSynchronize();
  perf_stats.vit_prefill_transition_time_ms = transition_timer.elapsed_ms();
  LOG(INFO) << "ViT->Prefill transition time: " << perf_stats.vit_prefill_transition_time_ms << " ms";
  
  // ========================================
  // Stage 3: Prefill
  // ========================================
  LOG(INFO) << "\n>>> Stage 3: Prefill <<<";
  LOG(INFO) << "Starting prefill with " << prefill_seq_len << " tokens...";
  inference::Timer prefill_timer;
  prefill_timer.start();
  
  status = model.prefill(embeddings, prefill_seq_len, 0);
  if (!status) {
    LOG(ERROR) << "Prefill failed: " << status.get_err_code();
    return 1;
  }
  
  // Sample first token
  int first_token = model.sample_first_token();
  
  cudaDeviceSynchronize();
  perf_stats.prefill_time_ms = prefill_timer.elapsed_ms();
  LOG(INFO) << "Prefill complete in " << perf_stats.prefill_time_ms << " ms";
  LOG(INFO) << "First token: " << first_token;
  
  if (first_token == vl_config.special_tokens.eos_token_id) {
    LOG(INFO) << "EOS token received, no generation needed";
    perf_stats.print();
    return 0;
  }
  
  // ========================================
  // Stage 4: Decode (Auto-regressive Generation)
  // ========================================
  LOG(INFO) << "\n>>> Stage 4: Decode (Auto-regressive Generation) <<<";
  inference::Timer decode_timer;
  decode_timer.start();
  
  std::vector<int32_t> generated;
  generated.push_back(static_cast<int32_t>(first_token));
  
  // Stream output: print first token immediately
  if (config.stream_output) {
    std::cout << "\n=== Response (Streaming) ===\n" << std::flush;
    std::string first_str = model.decode(static_cast<int32_t>(first_token));
    std::cout << first_str << std::flush;
  }
  
  int next_token = -1;
  for (int i = 1; i < config.max_tokens; ++i) {
    // Get embedding for the last generated token
    std::vector<int> last_token_vec = {generated.back()};
    auto embed_out = model.embedding(last_token_vec);
    
    // Decode step
    int pos = prefill_seq_len + static_cast<int>(generated.size()) - 1;
    status = model.decode_step(embed_out.input_embeddings, pos, next_token);
    if (!status) {
      LOG(ERROR) << "Decode step failed at pos=" << pos;
      break;
    }
    
    // Check for EOS
    if (next_token == vl_config.special_tokens.eos_token_id) {
      LOG(INFO) << "EOS token reached after " << generated.size() << " tokens";
      break;
    }
    
    generated.push_back(static_cast<int32_t>(next_token));
    
    // Stream output: print token immediately
    if (config.stream_output) {
      std::string token_str = model.decode(static_cast<int32_t>(next_token));
      std::cout << token_str << std::flush;
    }
    
    // Progress logging (only when not streaming to avoid mixing with output)
    if (!config.stream_output && (i + 1) % 50 == 0) {
      LOG(INFO) << "Generated " << generated.size() << " tokens...";
    }
  }
  
  cudaDeviceSynchronize();
  perf_stats.decode_time_ms = decode_timer.elapsed_ms();
  perf_stats.num_decode_tokens = static_cast<int>(generated.size());
  
  // Decode all tokens to get full response
  std::string response = model.decode(generated);
  
  // Print results
  if (config.stream_output) {
    std::cout << "\n================\n";
  } else {
    std::cout << "\n=== Response ===\n" << response << "\n================\n";
  }
  
  // Print performance statistics
  perf_stats.print();
  
  return 0;
}
#else
int run_inference(const VLInferenceConfig& config) {
  LOG(ERROR) << "Qwen3-VL support not enabled. Rebuild with -DQWEN3_VL_SUPPORT=ON";
  return 1;
}
#endif

int main(int argc, char* argv[]) {
  google::InitGoogleLogging(argv[0]);
  FLAGS_logtostderr = 1;
  
  std::cout << "Qwen3-VL Vision-Language Model Inference\n";
  std::cout << "=========================================\n";
  
  // Parse arguments
  VLInferenceConfig config = parse_args(argc, argv);
  
  // Validate
  if (!validate_config(config)) {
    print_usage(argv[0]);
    return 1;
  }
  
  // Set verbose logging
  if (config.verbose) {
    FLAGS_v = 1;
  }
  
  // Run inference
  return run_inference(config);
}
