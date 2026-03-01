#ifndef KUIPER_DEMO_INFERENCE_COMMON_H_
#define KUIPER_DEMO_INFERENCE_COMMON_H_

#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <functional>
#include <cuda_runtime.h>
#include <glog/logging.h>
#include "nlohmann/json.hpp"
#include "jinja.hpp"
#include <base/prefix_cache.h>
#include <base/alloc.h>
#include "model/model.h"

namespace inference {

// ==================== Qwen Chat Template ====================
// Qwen2/2.5/3 使用相同的 chat template

static const std::string QWEN_CHAT_TEMPLATE = R"(
{%- if tools %}
    {{- '<|im_start|>system\n' }}
    {%- if messages[0]['role'] == 'system' %}
        {{- messages[0]['content'] }}
    {%- else %}
        {{- 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.' }}
    {%- endif %}
    {{- "\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>" }}
    {%- for tool in tools %}
        {{- "\n" }}
        {{- tool | tojson }}
    {%- endfor %}
    {{- "\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n" }}
{%- else %}
    {%- if messages[0]['role'] == 'system' %}
        {{- '<|im_start|>system\n' + messages[0]['content'] + '<|im_end|>\n' }}
    {%- else %}
        {{- '<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n' }}
    {%- endif %}
{%- endif %}
{%- for message in messages %}
    {%- if (message.role == "user") or (message.role == "system" and not loop.first) or (message.role == "assistant" and not message.tool_calls) %}
        {{- '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>' + '\n' }}
    {%- elif message.role == "assistant" %}
        {{- '<|im_start|>' + message.role }}
        {%- if message.content %}
            {{- '\n' + message.content }}
        {%- endif %}
        {%- for tool_call in message.tool_calls %}
            {%- if tool_call.function is defined %}
                {%- set tool_call = tool_call.function %}
            {%- endif %}
            {{- '\n<tool_call>\n{"name": "' }}
            {{- tool_call.name }}
            {{- '", "arguments": ' }}
            {{- tool_call.arguments | tojson }}
            {{- '}\n</tool_call>' }}
        {%- endfor %}
        {{- '<|im_end|>\n' }}
    {%- elif message.role == "tool" %}
        {%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != "tool") %}
            {{- '<|im_start|>user' }}
        {%- endif %}
        {{- '\n<tool_response>\n' }}
        {{- message.content }}
        {{- '\n</tool_response>' }}
        {%- if loop.last or (messages[loop.index0 + 1].role != "tool") %}
            {{- '<|im_end|>\n' }}
        {%- endif %}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
{%- endif %}
)";

/**
 * @brief 应用 chat template 格式化消息
 */
inline std::string apply_chat_template(const nlohmann::json& messages, bool add_generation_prompt = true) {
    jinja::Template tpl(QWEN_CHAT_TEMPLATE);
    return tpl.apply_chat_template(messages, add_generation_prompt);
}

/**
 * @brief 从对话历史构建消息JSON
 */
inline nlohmann::json build_messages_json(
    const std::string& system_prompt,
    const std::vector<std::pair<std::string, std::string>>& history,
    const std::string& user_input) {
    
    nlohmann::json messages = nlohmann::json::array();
    
    // 添加系统消息
    messages.push_back({
        {"role", "system"},
        {"content", system_prompt}
    });
    
    // 添加历史对话
    for (const auto& [role, content] : history) {
        messages.push_back({
            {"role", role},
            {"content", content}
        });
    }
    
    // 添加当前用户输入
    if (!user_input.empty()) {
        messages.push_back({
            {"role", "user"},
            {"content", user_input}
        });
    }
    
    return messages;
}

// ==================== 推理配置 ====================

/**
 * @brief 推理配置参数
 */
struct InferenceConfig {
  bool use_cuda_graph = false;       // 是否使用CUDA Graph优化decode
  bool use_fused_ffn = true;         // 是否使用Fused FFN优化
  bool stream_output = false;        // 是否流式输出
  int max_tokens = 256;              // 最大生成token数
  int max_history_turns = 10;        // 最大保留的对话轮数
  bool interactive_mode = false;     // 是否交互式模式
  int max_context_len = 8192;        // 最大上下文长度
  bool use_prefix_cache = false;     // 是否使用 RadixTree prefix cache
  int64_t prefix_cache_size = 65536; // prefix cache 最大 token 数
  bool benchmark_mode = false;       // 是否运行性能基准测试
  int benchmark_decode_tokens = 1024; // benchmark模式下的decode token数
  base::AttentionType attention_type = base::AttentionType::kAttentionFlash1; // 注意力计算类型
  
  // 解析命令行参数
  static InferenceConfig parse_args(int argc, char* argv[], int start_idx = 3) {
    InferenceConfig config;
    
    for (int i = start_idx; i < argc; ++i) {
      std::string arg = argv[i];
      
      if (arg == "--cuda-graph") {
        config.use_cuda_graph = true;
      } else if (arg == "--no-cuda-graph") {
        config.use_cuda_graph = false;
      } else if (arg == "--no-fused-ffn") {
        config.use_fused_ffn = false;
      } else if (arg == "--stream") {
        config.stream_output = true;
      } else if (arg == "--interactive" || arg == "-i") {
        config.interactive_mode = true;
      } else if (arg == "--prefix-cache") {
        config.use_prefix_cache = true;
      } else if (arg == "--prefix-cache-size" && i + 1 < argc) {
        config.prefix_cache_size = std::atoll(argv[i + 1]);
        if (config.prefix_cache_size <= 0) {
          config.prefix_cache_size = 65536;
        }
        ++i;
      } else if (arg == "--max-tokens" && i + 1 < argc) {
        config.max_tokens = std::atoi(argv[i + 1]);
        if (config.max_tokens <= 0) {
          LOG(WARNING) << "Invalid --max-tokens value, using default 256";
          config.max_tokens = 256;
        }
        ++i;
      } else if (arg == "--attention" && i + 1 < argc) {
        std::string attn_type = argv[i + 1];
        if (attn_type == "mha") {
          config.attention_type = base::AttentionType::kAttentionMHA;
        } else if (attn_type == "flash1") {
          config.attention_type = base::AttentionType::kAttentionFlash1;
        } else if (attn_type == "flash2") {
          config.attention_type = base::AttentionType::kAttentionFlash2;
        } else {
          LOG(WARNING) << "Unknown attention type '" << attn_type 
                       << "', valid options: mha, flash1, flash2. Using default flash1";
        }
        ++i;
      } else if (arg == "--max-history" && i + 1 < argc) {
        config.max_history_turns = std::atoi(argv[i + 1]);
        if (config.max_history_turns < 0) {
          config.max_history_turns = 10;
        }
        ++i;
      } else if (arg == "--max-context" && i + 1 < argc) {
        config.max_context_len = std::atoi(argv[i + 1]);
        if (config.max_context_len <= 0) {
          config.max_context_len = 8192;
        }
        ++i;
      } else if (arg == "--benchmark") {
        config.benchmark_mode = true;
      } else if (arg == "--benchmark-decode-tokens" && i + 1 < argc) {
        config.benchmark_decode_tokens = std::atoi(argv[i + 1]);
        if (config.benchmark_decode_tokens <= 0) config.benchmark_decode_tokens = 1024;
        ++i;
      }
    }
    
    return config;
  }
  
  // 打印配置
  void print() const {
    LOG(INFO) << "=== Inference Configuration ===";
    LOG(INFO) << "CUDA Graph: " << (use_cuda_graph ? "enabled" : "disabled");
    LOG(INFO) << "Fused FFN: " << (use_fused_ffn ? "enabled" : "disabled");
    LOG(INFO) << "Attention: " << base::AttentionTypeName(attention_type);
    LOG(INFO) << "Stream output: " << (stream_output ? "enabled" : "disabled");
    LOG(INFO) << "Interactive mode: " << (interactive_mode ? "enabled" : "disabled");
    LOG(INFO) << "Prefix Cache: " << (use_prefix_cache ? "enabled" : "disabled");
    if (use_prefix_cache) {
      LOG(INFO) << "Prefix Cache Size: " << prefix_cache_size << " tokens";
    }
    LOG(INFO) << "Max tokens: " << max_tokens;
    LOG(INFO) << "Max context: " << max_context_len;
    if (interactive_mode) {
      LOG(INFO) << "Max history turns: " << max_history_turns;
    }
    LOG(INFO) << "===============================";
  }
};

/**
 * @brief 打印使用说明的通用函数
 */
inline void print_usage(const char* program_name, const std::string& model_desc) {
  LOG(INFO) << "Usage: " << program_name << " checkpoint_path tokenizer_path [options]";
  LOG(INFO) << "";
  LOG(INFO) << "Model: " << model_desc;
  LOG(INFO) << "";
  LOG(INFO) << "Options:";
  LOG(INFO) << "  --cuda-graph       Enable CUDA Graph for decode phase";
  LOG(INFO) << "  --no-cuda-graph    Disable CUDA Graph for decode phase";
  LOG(INFO) << "  --no-fused-ffn     Disable fused FFN optimization";
  LOG(INFO) << "  --attention TYPE   Set attention type: mha, flash1, flash2 (default: flash1)";
  LOG(INFO) << "  --stream           Enable streaming output (print tokens as generated)";
  LOG(INFO) << "  --interactive, -i  Enable interactive mode (continuous multi-turn dialog)";
  LOG(INFO) << "  --prefix-cache     Enable RadixTree-based prefix cache for KV cache reuse";
  LOG(INFO) << "  --prefix-cache-size N  Set maximum prefix cache size in tokens (default: 65536)";
  LOG(INFO) << "  --max-tokens N     Set maximum number of tokens to generate (default: 256)";
  LOG(INFO) << "  --max-history N    Set maximum history turns in interactive mode (default: 10)";
  LOG(INFO) << "  --max-context N    Set maximum context length (default: 8192)";
  LOG(INFO) << "";
  LOG(INFO) << "Interactive Commands:";
  LOG(INFO) << "  /clear             Clear conversation history and KV cache";
  LOG(INFO) << "  /stats             Show performance statistics";
  LOG(INFO) << "  /cache             Show prefix cache statistics (when --prefix-cache enabled)";
  LOG(INFO) << "  /help              Show this help";
  LOG(INFO) << "  /quit, /exit       Exit the program";
}

// ==================== 计时与性能统计 ====================

/**
 * @brief 采样策略：简单argmax
 */
inline int32_t sample_argmax(const std::vector<float>& logits) {
  int32_t best_idx = 0;
  float max_val = logits[0];
  for (size_t i = 1; i < logits.size(); ++i) {
    if (logits[i] > max_val) {
      max_val = logits[i];
      best_idx = static_cast<int32_t>(i);
    }
  }
  return best_idx;
}

/**
 * @brief 计时器类，用于性能统计
 */
class Timer {
 public:
  void start() {
    start_time_ = std::chrono::steady_clock::now();
  }
  
  double elapsed_ms() const {
    auto end_time = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time_).count() / 1000.0;
  }
  
  double elapsed_us() const {
    auto end_time = std::chrono::steady_clock::now();
    return static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time_).count());
  }
  
 private:
  std::chrono::steady_clock::time_point start_time_;
};

/**
 * @brief 性能统计结构
 * 
 * 关于 prefill/decode 速度随上下文变长而下降的说明：
 * ============================================
 * 这是 Transformer Attention 机制的正常行为，不是 bug。
 * 
 * 原因：
 * - Prefill: 每个新 token 需要 attention 到 [0, start_pos + seq_idx] 范围的所有 KV
 * - Decode: 每个新 token 需要 attention 到 [0, pos] 范围的所有 KV
 * 
 * 计算复杂度：
 * - Prefill: O(prefill_tokens × total_context_length)
 * - Decode: O(1 × current_position)
 * 
 * 所以随着对话进行，上下文变长，每个 token 的计算量会线性增加。
 * 这就是为什么 KV cache 复用虽然减少了 prefill 的 token 数，
 * 但每个 token 的计算时间会随着历史变长而增加。
 * 
 * 性能指标解读：
 * - prefill_throughput: 实际 prefill 的吞吐量（会随上下文变长而下降）
 * - attention_compute: 实际 attention 计算量 = prefill_tokens × avg_attention_len
 * - decode_throughput: decode 吞吐量（会随上下文变长而下降）
 */
struct PerfStats {
  int32_t prompt_len = 0;           // prompt长度
  int32_t prefill_tokens = 0;       // 实际prefill的token数
  int32_t decode_steps = 0;         // decode步数
  double prefill_time_ms = 0.0;     // prefill时间(ms)
  double decode_time_ms = 0.0;      // decode时间(ms)
  int32_t kv_reuse_len = 0;         // KV cache复用长度
  int32_t total_context_len = 0;    // 总上下文长度（用于计算 attention 范围）
  
  // 累计统计
  int64_t total_prefill_tokens = 0;
  int64_t total_decode_tokens = 0;
  double total_prefill_time_ms = 0.0;
  double total_decode_time_ms = 0.0;
  int32_t request_count = 0;
  
  double prefill_throughput() const {
    return prefill_time_ms > 0 ? (prefill_tokens * 1000.0 / prefill_time_ms) : 0;
  }
  
  double decode_throughput() const {
    return decode_time_ms > 0 ? (decode_steps * 1000.0 / decode_time_ms) : 0;
  }
  
  // 计算 attention 计算量（考虑上下文长度的影响）
  // attention_compute = sum(attention_len[i]) for i in prefill_tokens
  // 对于增量 prefill: 每个 token i 的 attention 范围是 [0, kv_reuse_len + i]
  // 平均 attention 长度 ≈ kv_reuse_len + prefill_tokens/2
  double avg_attention_len() const {
    if (prefill_tokens == 0) return 0;
    return kv_reuse_len + (prefill_tokens + 1) / 2.0;
  }
  
  // 有效 prefill 吞吐量（归一化到 attention 长度=1024 的基准）
  // 这个指标消除了上下文长度的影响，更能反映实际的计算效率
  double normalized_prefill_throughput() const {
    double raw_tp = prefill_throughput();
    double avg_attn = avg_attention_len();
    if (avg_attn <= 0) return raw_tp;
    // 归一化到 1024 token 的 attention 范围
    return raw_tp * (avg_attn / 1024.0);
  }
  
  // 累计吞吐量
  double avg_prefill_throughput() const {
    return total_prefill_time_ms > 0 ? (total_prefill_tokens * 1000.0 / total_prefill_time_ms) : 0;
  }
  
  double avg_decode_throughput() const {
    return total_decode_time_ms > 0 ? (total_decode_tokens * 1000.0 / total_decode_time_ms) : 0;
  }
  
  void accumulate() {
    total_prefill_tokens += prefill_tokens;
    total_decode_tokens += decode_steps;
    total_prefill_time_ms += prefill_time_ms;
    total_decode_time_ms += decode_time_ms;
    request_count++;
  }
  
  void print(bool verbose = true) const {
    if (verbose) {
      LOG(INFO) << "\n=== Performance Statistics ===";
      LOG(INFO) << "Prompt length: " << prompt_len << " tokens";
      if (kv_reuse_len > 0) {
        LOG(INFO) << "KV cache reuse: " << kv_reuse_len << " tokens (" 
                  << (kv_reuse_len * 100 / prompt_len) << "%)";
      }
      LOG(INFO) << "Prefill: " << prefill_tokens << " tokens, " 
                << prefill_time_ms << " ms, " 
                << prefill_throughput() << " tokens/s";
      if (kv_reuse_len > 0) {
        LOG(INFO) << "  (avg attention range: " << (int)avg_attention_len() 
                  << ", normalized throughput: " << normalized_prefill_throughput() << " tokens/s)";
      }
      LOG(INFO) << "Decode: " << decode_steps << " tokens, " 
                << decode_time_ms << " ms, " 
                << decode_throughput() << " tokens/s";
      int32_t final_ctx_len = prompt_len + decode_steps;
      LOG(INFO) << "  (final context: " << final_ctx_len << " tokens)";
      LOG(INFO) << "===============================";
    } else {
      // 紧凑格式（用于stream模式）
      std::cerr << "\n[Prefill: " << prefill_tokens << " tokens, " 
                << prefill_throughput() << " tokens/s";
      if (kv_reuse_len > 0) {
        std::cerr << " (KV reuse: " << kv_reuse_len << ", attn: " << (int)avg_attention_len() << ")";
      }
      std::cerr << " | Decode: " << decode_steps << " tokens, " 
                << decode_throughput() << " tokens/s";
      int32_t final_ctx_len = prompt_len + decode_steps;
      std::cerr << " (ctx: " << final_ctx_len << ")]" << std::endl;
    }
  }
  
  void print_cumulative() const {
    LOG(INFO) << "\n=== Cumulative Statistics ===";
    LOG(INFO) << "Total requests: " << request_count;
    LOG(INFO) << "Total prefill: " << total_prefill_tokens << " tokens, " 
              << total_prefill_time_ms << " ms, "
              << avg_prefill_throughput() << " tokens/s avg";
    LOG(INFO) << "Total decode: " << total_decode_tokens << " tokens, " 
              << total_decode_time_ms << " ms, "
              << avg_decode_throughput() << " tokens/s avg";
    LOG(INFO) << "==============================";
  }
};

// ==================== 多轮对话管理 ====================

/**
 * @brief 多轮对话管理器
 * 
 * 设计原理 (参考 llama.cpp):
 * - 维护对话历史和当前KV cache位置
 * - 使用 token 级别的缓存确保一致性
 * - 每轮对话时计算增量tokens，只prefill新增部分
 * - 支持上下文溢出时的历史截断
 */
class MultiTurnConversation {
 public:
  MultiTurnConversation(const std::string& system_prompt = 
      "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.")
      : system_prompt_(system_prompt) {}
  
  /**
   * @brief 添加用户消息到历史
   */
  void add_user_message(const std::string& content) {
    history_.push_back({"user", content});
  }
  
  /**
   * @brief 添加助手消息到历史
   */
  void add_assistant_message(const std::string& content) {
    history_.push_back({"assistant", content});
  }
  
  /**
   * @brief 获取完整的格式化prompt（包含所有历史+当前输入）
   */
  std::string get_full_prompt(const std::string& user_input) const {
    nlohmann::json messages = build_messages_json(system_prompt_, history_, user_input);
    return apply_chat_template(messages, true);
  }
  
  /**
   * @brief 获取历史 prompt（不包含新的用户输入，不添加 generation prompt）
   * 
   * 用于在每轮对话结束后重新 tokenize 历史，确保 cached_tokens_ 与下一轮
   * tokenize 结果的前缀完全一致，从而实现 KV cache 复用。
   */
  std::string get_history_prompt() const {
    nlohmann::json messages = nlohmann::json::array();
    
    // 添加系统消息
    messages.push_back({
        {"role", "system"},
        {"content", system_prompt_}
    });
    
    // 添加历史对话
    for (const auto& [role, content] : history_) {
        messages.push_back({
            {"role", role},
            {"content", content}
        });
    }
    
    // 不添加 generation prompt，因为这是完整的历史
    return apply_chat_template(messages, false);
  }
  
  /**
   * @brief 获取当前KV cache位置（已处理的token数）
   */
  int32_t get_kv_position() const {
    return current_kv_pos_;
  }
  
  /**
   * @brief 设置当前KV cache位置
   */
  void set_kv_position(int32_t pos) {
    current_kv_pos_ = pos;
  }
  
  /**
   * @brief 更新已缓存的token序列（用于增量计算）
   */
  void update_cached_tokens(const std::vector<int32_t>& tokens) {
    cached_tokens_ = tokens;
    current_kv_pos_ = static_cast<int32_t>(tokens.size());
  }
  
  /**
   * @brief 追加单个token到缓存（decode时调用）
   */
  void append_token(int32_t token) {
    cached_tokens_.push_back(token);
    current_kv_pos_++;
  }
  
  /**
   * @brief 获取已缓存的tokens
   */
  const std::vector<int32_t>& get_cached_tokens() const {
    return cached_tokens_;
  }
  
  /**
   * @brief 计算与当前tokens的公共前缀长度
   * 
   * 重要：返回值必须与 current_kv_pos_ 比较，
   * 如果不相等说明缓存无效需要重新prefill
   */
  int32_t compute_common_prefix_len(const std::vector<int32_t>& new_tokens) const {
    int32_t common_len = 0;
    size_t max_compare = std::min(cached_tokens_.size(), new_tokens.size());
    for (size_t i = 0; i < max_compare; ++i) {
      if (cached_tokens_[i] == new_tokens[i]) {
        ++common_len;
      } else {
        break;
      }
    }
    return common_len;
  }
  
  /**
   * @brief 同步缓存的tokens与实际tokenize结果
   * 
   * 在每轮对话结束后调用，确保 cached_tokens_ 与
   * 下一轮 tokenize 的结果前缀部分一致
   * 
   * @param full_history_tokens 完整历史的 tokens（通过重新 tokenize 获得）
   */
  void sync_cached_tokens(const std::vector<int32_t>& full_history_tokens) {
    // 只有当缓存不为空时才同步
    if (cached_tokens_.empty()) {
      return;
    }
    
    // 检查是否匹配
    int32_t common = compute_common_prefix_len(full_history_tokens);
    if (common == current_kv_pos_) {
      // 完全匹配，更新缓存为新的 tokens
      cached_tokens_ = full_history_tokens;
      current_kv_pos_ = static_cast<int32_t>(full_history_tokens.size());
    }
    // 如果不匹配，保持原样，下一轮会检测到不匹配并清空
  }
  
  /**
   * @brief 清空对话历史和KV cache状态
   */
  void clear() {
    history_.clear();
    cached_tokens_.clear();
    current_kv_pos_ = 0;
  }
  
  /**
   * @brief 重置KV cache状态（保留对话历史）
   * 当检测到 tokenization 不一致时调用
   */
  void reset_kv_state() {
    cached_tokens_.clear();
    current_kv_pos_ = 0;
  }
  
  /**
   * @brief 重置（清空历史但保留系统提示）
   */
  void reset() {
    clear();
  }
  
  /**
   * @brief 截断历史（保留最近N轮对话）
   * 截断后需要重新从头构建KV cache
   * @return true 如果发生了截断
   */
  bool truncate_history(size_t max_turns) {
    size_t max_messages = max_turns * 2;  // 每轮包含user和assistant
    if (history_.size() > max_messages) {
      history_.erase(history_.begin(), history_.end() - max_messages);
      // 截断后KV cache无效，需要重建
      cached_tokens_.clear();
      current_kv_pos_ = 0;
      return true;  // 表示发生了截断
    }
    return false;
  }
  
  /**
   * @brief 设置系统提示
   */
  void set_system_prompt(const std::string& prompt) {
    system_prompt_ = prompt;
  }
  
  /**
   * @brief 获取对话轮数
   */
  size_t get_turn_count() const {
    return history_.size() / 2;
  }
  
  /**
   * @brief 获取历史消息
   */
  const std::vector<std::pair<std::string, std::string>>& get_history() const {
    return history_;
  }
  
 private:
  std::string system_prompt_;
  std::vector<std::pair<std::string, std::string>> history_;
  std::vector<int32_t> cached_tokens_;  // 已在KV cache中的tokens
  int32_t current_kv_pos_ = 0;          // 当前KV cache位置
};

// ==================== Qwen3 Thinking 处理 ====================

/**
 * @brief 从 Qwen3 模型的回复中移除 <think>...</think> 部分
 * 如果没有完整的 </think> 结束标记，返回完整内容（让用户看到思考过程）
 */
inline std::string remove_thinking_content(const std::string& response) {
    const std::string think_start = "<think>";
    const std::string think_end = "</think>";
    
    std::string result = response;
    
    size_t end_pos = result.find(think_end);
    if (end_pos != std::string::npos) {
        // 找到了完整的 </think>，移除思考部分
        size_t content_start = end_pos + think_end.length();
        result = result.substr(content_start);
        
        // 去除开头的空白
        size_t first_non_space = result.find_first_not_of(" \t\n\r");
        if (first_non_space != std::string::npos) {
            result = result.substr(first_non_space);
        } else {
            result.clear();
        }
    }
    // 如果没有 </think>，保留原始内容（包括 <think> 标签和思考内容）
    // 这样用户可以看到不完整的回复
    
    return result;
}

// ==================== 命令处理 ====================

/**
 * @brief 处理交互式命令
 * @return true 如果命令已处理，false 如果是普通输入
 */
inline bool process_command(const std::string& input, 
                           MultiTurnConversation& conv,
                           PerfStats& cumulative_stats,
                           std::function<void()> clear_kv_cache_fn) {
    if (input.empty()) return true;
    
    if (input == "/quit" || input == "/exit") {
        LOG(INFO) << "Goodbye!";
        exit(0);
    }
    
    if (input == "/help") {
        LOG(INFO) << "\nAvailable commands:";
        LOG(INFO) << "  /clear   - Clear conversation history and KV cache";
        LOG(INFO) << "  /stats   - Show cumulative statistics";
        LOG(INFO) << "  /quit    - Exit the program";
        return true;
    }
    
    if (input == "/clear") {
        conv.clear();
        clear_kv_cache_fn();
        LOG(INFO) << "Conversation history and KV cache cleared.";
        return true;
    }
    
    if (input == "/stats") {
        cumulative_stats.print_cumulative();
        return true;
    }
    
    // 不是命令
    if (input[0] == '/') {
        LOG(INFO) << "Unknown command: " << input;
        LOG(INFO) << "Type /help for available commands.";
        return true;
    }
    
    return false;
}

/**
 * @brief 读取用户输入
 */
inline std::string read_user_input() {
    std::string line;
    
    std::cout << "\n>>> ";
    std::cout.flush();
    
    if (!std::getline(std::cin, line)) {
        return "/quit";
    }
    
    return line;
}

// ==================== RadixTree PrefixCache 管理 ====================

/**
 * @brief PrefixCache 管理器
 * 
 * 提供基于 RadixTree 的完整 prefix cache 功能:
 * 1. 使用 RadixTree 存储 token 序列到 KV Cache 的映射
 * 2. 支持跨请求的前缀共享
 * 3. 提供 LRU 淘汰策略
 * 
 * 与 MultiTurnConversation 的简单前缀匹配相比:
 * - MultiTurnConversation: 只在单个对话内复用 KV cache，线性匹配
 * - PrefixCacheManager: 支持多个对话间的前缀共享，树结构高效匹配
 */
class PrefixCacheManager {
 public:
    explicit PrefixCacheManager(int64_t max_tokens = 65536) {
        base::PrefixCacheConfig config;
        config.max_cached_tokens = max_tokens;
        config.min_prefix_length = 4;
        config.enable_auto_eviction = true;
        config.eviction_threshold = 0.9f;
        config.enable_stats = true;
        prefix_cache_ = std::make_unique<base::PrefixCache>(config);
    }
    
    /**
     * @brief 查找前缀匹配
     * 
     * @param tokens 新请求的 token 序列
     * @return 匹配结果（可复用的 token 数、需要计算的起始位置等）
     */
    base::PrefixMatchResult match(const std::vector<int32_t>& tokens) {
        return prefix_cache_->match(tokens);
    }
    
    /**
     * @brief 注册新的前缀
     * 
     * 在 prefill 完成后调用，将新计算的序列注册到缓存
     */
    void register_prefix(const std::vector<int32_t>& tokens, int32_t kv_length) {
        prefix_cache_->insert(tokens, 0, kv_length);
    }
    
    /**
     * @brief 更新前缀（decode 阶段逐步扩展）
     * 
     * 在 decode 过程中，每生成一个 token 就更新缓存
     */
    void extend_prefix(const std::vector<int32_t>& tokens, int32_t new_kv_length) {
        // 插入扩展后的序列
        prefix_cache_->insert(tokens, 0, new_kv_length);
    }
    
    /**
     * @brief 释放前缀引用
     */
    void release(const std::vector<int32_t>& tokens) {
        prefix_cache_->release(tokens);
    }
    
    /**
     * @brief 清空缓存
     */
    void clear() {
        prefix_cache_->clear();
    }
    
    /**
     * @brief 获取统计信息
     */
    const base::PrefixCacheStats& get_stats() const {
        return prefix_cache_->get_stats();
    }
    
    /**
     * @brief 打印统计信息
     */
    void print_stats() const {
        LOG(INFO) << prefix_cache_->get_stats().to_string();
        auto tree_stats = prefix_cache_->get_tree_stats();
        LOG(INFO) << "RadixTree Stats:";
        LOG(INFO) << "  Total nodes: " << tree_stats.total_nodes;
        LOG(INFO) << "  Terminal nodes: " << tree_stats.terminal_nodes;
        LOG(INFO) << "  Max depth: " << tree_stats.max_depth;
        LOG(INFO) << "  Cached tokens: " << tree_stats.total_cached_tokens;
    }
    
    /**
     * @brief 获取缓存的 token 数
     */
    int64_t get_cached_tokens() const {
        return prefix_cache_->get_cached_tokens();
    }
    
    /**
     * @brief 检查是否启用
     */
    bool is_enabled() const {
        return enabled_;
    }
    
    void set_enabled(bool enabled) {
        enabled_ = enabled;
    }
    
 private:
    std::unique_ptr<base::PrefixCache> prefix_cache_;
    bool enabled_ = true;
};

/**
 * @brief 扩展的多轮对话管理器（支持 RadixTree PrefixCache）
 * 
 * 在原有 MultiTurnConversation 基础上增加 PrefixCache 支持:
 * - 当 PrefixCache 启用时，使用 RadixTree 进行前缀匹配
 * - 否则回退到原有的线性前缀匹配
 */
class MultiTurnConversationWithCache : public MultiTurnConversation {
 public:
    MultiTurnConversationWithCache(
        const std::string& system_prompt = 
            "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
        PrefixCacheManager* cache_manager = nullptr)
        : MultiTurnConversation(system_prompt), cache_manager_(cache_manager) {}
    
    /**
     * @brief 查找前缀匹配（优先使用 PrefixCache）
     * 
     * @param new_tokens 新请求的 token 序列
     * @return pair<匹配长度, 是否使用了 PrefixCache>
     */
    std::pair<int32_t, bool> find_prefix_match(const std::vector<int32_t>& new_tokens) {
        if (cache_manager_ && cache_manager_->is_enabled()) {
            // 使用 RadixTree 前缀匹配
            auto result = cache_manager_->match(new_tokens);
            if (result.cache_hit) {
                last_matched_prefix_ = result.matched_prefix;
                return {result.matched_tokens, true};
            }
        }
        
        // 回退到原有的线性前缀匹配
        int32_t common_len = compute_common_prefix_len(new_tokens);
        int32_t cached_len = static_cast<int32_t>(get_cached_tokens().size());
        
        if (cached_len > 0 && common_len == cached_len && common_len > 0) {
            return {common_len, false};
        }
        
        return {0, false};
    }
    
    /**
     * @brief 注册前缀到缓存
     */
    void register_to_cache(const std::vector<int32_t>& tokens) {
        if (cache_manager_ && cache_manager_->is_enabled()) {
            cache_manager_->register_prefix(tokens, static_cast<int32_t>(tokens.size()));
        }
    }
    
    /**
     * @brief 释放前缀引用
     */
    void release_cache_reference() {
        if (cache_manager_ && cache_manager_->is_enabled() && !last_matched_prefix_.empty()) {
            cache_manager_->release(last_matched_prefix_);
            last_matched_prefix_.clear();
        }
    }
    
    /**
     * @brief 获取 PrefixCache 管理器
     */
    PrefixCacheManager* get_cache_manager() {
        return cache_manager_;
    }
    
    void set_cache_manager(PrefixCacheManager* manager) {
        cache_manager_ = manager;
    }
    
 private:
    PrefixCacheManager* cache_manager_ = nullptr;
    std::vector<int32_t> last_matched_prefix_;
};

/**
 * @brief 处理交互式命令（扩展版，支持 PrefixCache 命令）
 * @return true 如果命令已处理，false 如果是普通输入
 */
inline bool process_command_with_cache(
    const std::string& input, 
    MultiTurnConversation& conv,
    PerfStats& cumulative_stats,
    std::function<void()> clear_kv_cache_fn,
    PrefixCacheManager* cache_manager = nullptr) {
    
    if (input.empty()) return true;
    
    if (input == "/quit" || input == "/exit") {
        LOG(INFO) << "Goodbye!";
        exit(0);
    }
    
    if (input == "/help") {
        LOG(INFO) << "\nAvailable commands:";
        LOG(INFO) << "  /clear   - Clear conversation history and KV cache";
        LOG(INFO) << "  /stats   - Show cumulative statistics";
        if (cache_manager) {
            LOG(INFO) << "  /cache   - Show prefix cache statistics";
        }
        LOG(INFO) << "  /quit    - Exit the program";
        return true;
    }
    
    if (input == "/clear") {
        conv.clear();
        clear_kv_cache_fn();
        if (cache_manager) {
            cache_manager->clear();
            LOG(INFO) << "Conversation history, KV cache, and prefix cache cleared.";
        } else {
            LOG(INFO) << "Conversation history and KV cache cleared.";
        }
        return true;
    }
    
    if (input == "/stats") {
        cumulative_stats.print_cumulative();
        return true;
    }
    
    if (input == "/cache") {
        if (cache_manager) {
            cache_manager->print_stats();
        } else {
            LOG(INFO) << "Prefix cache is not enabled. Use --prefix-cache to enable.";
        }
        return true;
    }
    
    // 不是命令
    if (input[0] == '/') {
        LOG(INFO) << "Unknown command: " << input;
        LOG(INFO) << "Type /help for available commands.";
        return true;
    }
    
    return false;
}

// ==================== 模型推理配置 ====================

/**
 * @brief 模型推理行为配置
 * 
 * 用于参数化不同模型（Qwen2/Qwen3等）在推理时的差异行为：
 * - skip_tokens: 在输出中跳过的特殊token（如EOS/BOS）
 * - remove_thinking: 是否移除 <think>...</think> 思考内容
 * - model_name: 模型显示名称
 */
struct ModelInferConfig {
    std::vector<int32_t> skip_tokens;   // 推理输出中需要跳过的token ID
    bool remove_thinking = false;        // 是否移除thinking内容（Qwen3特性）
    std::string model_name;              // 显示名称（如 "Qwen2/2.5", "Qwen3"）
    
    /**
     * @brief 判断token是否应该跳过
     */
    bool should_skip(int32_t token) const {
        for (auto t : skip_tokens) {
            if (t == token) return true;
        }
        return false;
    }
    
    /**
     * @brief 可选地移除thinking内容
     */
    std::string post_process(const std::string& response) const {
        if (remove_thinking) {
            return remove_thinking_content(response);
        }
        return response;
    }
};

// ==================== 模板化推理核心函数 ====================

/**
 * @brief 执行一轮生成（支持增量prefill和RadixTree PrefixCache）
 * 
 * 模板化的通用生成函数，适用于所有 Qwen 系列模型。
 * 
 * @tparam ModelType 模型类型（如 model::Qwen2Model, model::Qwen3Model）
 * @param model 模型实例
 * @param conv 对话管理器
 * @param user_input 用户输入
 * @param config 推理配置
 * @param stats 性能统计
 * @param cache_manager RadixTree PrefixCache 管理器（可为nullptr）
 * @param model_config 模型行为配置
 * @return 生成的回复
 */
template<typename ModelType>
std::string generate_response(
    ModelType& model,
    MultiTurnConversation& conv,
    const std::string& user_input,
    const InferenceConfig& config,
    PerfStats& stats,
    PrefixCacheManager* cache_manager,
    const ModelInferConfig& model_config) {
    
    // 1. 获取完整的格式化prompt
    std::string full_prompt = conv.get_full_prompt(user_input);
    
    // 2. Tokenize
    auto tokens = model.encode(full_prompt);
    std::vector<int32_t> tokens_i32(tokens.begin(), tokens.end());
    int32_t total_len = static_cast<int32_t>(tokens_i32.size());
    
    stats.prompt_len = total_len;
    
    // 检查是否超过最大上下文长度
    int32_t max_seq_len = model.get_config()->seq_len_;
    if (total_len + config.max_tokens > max_seq_len) {
        LOG(WARNING) << "Context length (" << total_len << " + " << config.max_tokens 
                     << " = " << (total_len + config.max_tokens)
                     << ") exceeds max_seq_len (" << max_seq_len 
                     << "). Clearing KV cache and truncating history.";
        return "[Context length exceeded. Please use /clear to start a new conversation or reduce history.]";
    }
    
    // 3. 计算与已缓存tokens的公共前缀长度
    int32_t start_pos = 0;
    bool used_radix_cache = false;
    std::vector<int32_t> matched_prefix;
    
    if (config.use_prefix_cache && cache_manager) {
        auto match_result = cache_manager->match(tokens_i32);
        if (match_result.cache_hit && match_result.matched_tokens > 0) {
            start_pos = match_result.matched_tokens;
            stats.kv_reuse_len = match_result.matched_tokens;
            used_radix_cache = true;
            matched_prefix = match_result.matched_prefix;
            
            if (!config.stream_output) {
                LOG(INFO) << "[RadixTree] Cache hit! Reusing " << match_result.matched_tokens 
                          << "/" << total_len << " tokens (" 
                          << (int)(match_result.reuse_ratio * 100) << "%)";
            }
        } else {
            model.clear_kv_cache();
            conv.reset_kv_state();
            stats.kv_reuse_len = 0;
        }
    } else {
        int32_t common_prefix_len = conv.compute_common_prefix_len(tokens_i32);
        int32_t cached_len = static_cast<int32_t>(conv.get_cached_tokens().size());
        
        if (cached_len > 0 && common_prefix_len == cached_len && common_prefix_len > 0) {
            start_pos = common_prefix_len;
            stats.kv_reuse_len = common_prefix_len;
        } else if (cached_len > 0 && common_prefix_len < cached_len) {
            model.clear_kv_cache();
            conv.reset_kv_state();
            stats.kv_reuse_len = 0;
        } else {
            stats.kv_reuse_len = 0;
        }
    }
    
    stats.prefill_tokens = total_len - start_pos;
    
    if (!config.stream_output && stats.kv_reuse_len > 0 && !used_radix_cache) {
        LOG(INFO) << "KV cache reuse: " << stats.kv_reuse_len << "/" << total_len 
                  << " tokens (" << (stats.kv_reuse_len * 100 / total_len) << "%)";
    }
    
    // 4-5. 获取embeddings并Prefill
    // 优化：只为需要prefill的新token计算embedding，避免对已在KV cache中的token做多余计算
    Timer prefill_timer;
    prefill_timer.start();
    
    if (stats.prefill_tokens > 0) {
        if (start_pos > 0) {
            // 增量prefill: 只对新token(tokens[start_pos:])计算embedding
            std::vector<int> new_tokens(tokens.begin() + start_pos, tokens.end());
            const auto& embedding_out = model.embedding(new_tokens);
            
            base::Status status = model.prefill(embedding_out.input_embeddings, stats.prefill_tokens, start_pos);
            if (!status) {
                LOG(ERROR) << "Incremental prefill failed: " << status.get_err_code();
                return "";
            }
        } else {
            // 全量prefill: 计算所有token的embedding
            const auto& embedding_out = model.embedding(tokens);
            
            base::Status status = model.prefill(embedding_out.input_embeddings, total_len, 0);
            if (!status) {
                LOG(ERROR) << "Prefill failed: " << status.get_err_code();
                return "";
            }
        }
    }
    
    if (model.get_cuda_config()) {
        cudaStreamSynchronize(model.get_cuda_config()->stream);
    }
    
    stats.prefill_time_ms = prefill_timer.elapsed_ms();
    
    // 6. 更新缓存的tokens
    if (start_pos > 0 && !used_radix_cache) {
        const auto& cached = conv.get_cached_tokens();
        std::vector<int32_t> new_cached(cached.begin(), cached.begin() + start_pos);
        new_cached.insert(new_cached.end(), tokens_i32.begin() + start_pos, tokens_i32.end());
        conv.update_cached_tokens(new_cached);
    } else {
        conv.update_cached_tokens(tokens_i32);
    }
    
    // 7. 采样第一个token
    tensor::Tensor forward_output = model.get_buffer(model::ModelBufferType::kForwardOutput);
    std::vector<float> logits_cpu(forward_output.size());
    cudaMemcpy(logits_cpu.data(), forward_output.template ptr<float>(),
               forward_output.size() * sizeof(float), cudaMemcpyDeviceToHost);
    
    int32_t next = sample_argmax(logits_cpu);
    
    // 8. Decode循环
    Timer decode_timer;
    decode_timer.start();
    
    std::vector<int32_t> generated_tokens;
    std::string prev_decoded_text;
    int32_t pos = total_len;
    int decode_steps = 0;
    
    // 首个token处理（跳过配置的特殊tokens）
    if (!model_config.should_skip(next)) {
        generated_tokens.push_back(next);
        if (config.stream_output) {
            std::string decoded = static_cast<model::Model&>(model).decode(generated_tokens);
            std::string new_text = decoded.substr(prev_decoded_text.length());
            if (!new_text.empty()) {
                printf("%s", new_text.c_str());
                fflush(stdout);
            }
            prev_decoded_text = decoded;
        }
    }
    
    conv.append_token(next);
    
    while (decode_steps < config.max_tokens) {
        if (model.is_sentence_ending(next)) {
            break;
        }
        
        std::vector<int32_t> single_token = {next};
        const auto& token_embedding = model.embedding(single_token);
        
        tensor::Tensor pos_tensor = model.get_buffer(model::ModelBufferType::kInputPos);
        pos_tensor.index<int32_t>(0) = pos;
        tensor::Tensor input = model.fill_input(pos_tensor, token_embedding, false);
        
        auto decode_status = model.decode(input, pos, next);
        if (!decode_status) {
            LOG(ERROR) << "Decode failed: " << decode_status.get_err_code();
            break;
        }
        
        conv.append_token(next);
        
        // 收集生成的token（跳过配置的特殊tokens）
        if (!model_config.should_skip(next)) {
            generated_tokens.push_back(next);
            if (config.stream_output) {
                std::string decoded = static_cast<model::Model&>(model).decode(generated_tokens);
                std::string new_text = decoded.substr(prev_decoded_text.length());
                if (!new_text.empty()) {
                    printf("%s", new_text.c_str());
                    fflush(stdout);
                }
                prev_decoded_text = decoded;
            }
        }
        
        pos++;
        decode_steps++;
    }
    
    if (model.get_cuda_config()) {
        cudaStreamSynchronize(model.get_cuda_config()->stream);
    }
    
    stats.decode_time_ms = decode_timer.elapsed_ms();
    stats.decode_steps = decode_steps;
    
    // 9. 注册到 RadixTree PrefixCache
    if (config.use_prefix_cache && cache_manager) {
        std::vector<int32_t> final_tokens = conv.get_cached_tokens();
        cache_manager->register_prefix(final_tokens, static_cast<int32_t>(final_tokens.size()));
        
        if (!matched_prefix.empty()) {
            cache_manager->release(matched_prefix);
        }
    }
    
    if (generated_tokens.empty()) {
        return "";
    }
    return static_cast<model::Model&>(model).decode(generated_tokens);
}

/**
 * @brief 运行交互式多轮对话（支持 RadixTree PrefixCache）
 * 
 * @tparam ModelType 模型类型
 */
template<typename ModelType>
void run_interactive(
    ModelType& model,
    const InferenceConfig& config,
    PrefixCacheManager* cache_manager,
    const ModelInferConfig& model_config) {
    
    MultiTurnConversation conv;
    PerfStats cumulative_stats;
    
    LOG(INFO) << "\n=== Interactive Multi-Turn Dialog (" << model_config.model_name << ") ===";
    LOG(INFO) << "Type your message and press Enter.";
    if (config.use_prefix_cache) {
        LOG(INFO) << "[RadixTree PrefixCache ENABLED]";
        LOG(INFO) << "Commands: /help, /clear, /stats, /cache, /quit";
    } else {
        LOG(INFO) << "Commands: /help, /clear, /stats, /quit";
    }
    LOG(INFO) << "================================================\n";
    
    auto clear_kv_cache_fn = [&model, cache_manager]() {
        model.clear_kv_cache();
        if (cache_manager) {
            cache_manager->clear();
        }
    };
    
    while (true) {
        std::string user_input = read_user_input();
        
        if (process_command_with_cache(user_input, conv, cumulative_stats, clear_kv_cache_fn, 
                                       cache_manager)) {
            continue;
        }
        
        if (conv.truncate_history(config.max_history_turns)) {
            LOG(INFO) << "[History truncated to " << config.max_history_turns << " turns]";
            model.clear_kv_cache();
            if (cache_manager) {
                cache_manager->clear();
            }
        }
        
        PerfStats stats;
        std::string response = generate_response(model, conv, user_input, config, stats,
                                                 cache_manager, model_config);
        
        if (!response.empty()) {
            std::string display_response = model_config.post_process(response);
            
            if (!config.stream_output) {
                LOG(INFO) << "\nAssistant: " << display_response;
            } else {
                std::cout << std::endl;
            }
            
            // 添加到对话历史（存储完整回复，包含thinking内容，以保持KV cache一致性）
            conv.add_user_message(user_input);
            conv.add_assistant_message(response);
            
            // 重新 tokenize 完整历史来同步 cached_tokens_
            std::string history_prompt = conv.get_history_prompt();
            auto history_tokens = model.encode(history_prompt);
            std::vector<int32_t> history_tokens_i32(history_tokens.begin(), history_tokens.end());
            
            // 修复 KV cache 与 cached_tokens 不一致的问题：
            // 模型 decode 时，最后输出的 token（如 <|im_end|>）不会被写入 KV cache
            // （因为 EOS 检测后直接 break，该 token 从未作为输入经过 decode）。
            // 此外 chat template 可能在末尾追加 \n 等分隔符 token。
            // 这些 token 出现在重新 tokenize 的历史中，但对应的 KV cache 位置未填充。
            // 如果不补齐，下一轮对话的 prefix cache 会声称复用这些位置，
            // 导致 attention 计算使用未初始化的 KV 数据（正确性 bug）。
            int32_t actual_kv_len = stats.prompt_len + stats.decode_steps;
            int32_t retokenized_len = static_cast<int32_t>(history_tokens_i32.size());
            
            if (retokenized_len > actual_kv_len) {
                int32_t gap = retokenized_len - actual_kv_len;
                std::vector<int> gap_tokens(history_tokens_i32.begin() + actual_kv_len,
                                            history_tokens_i32.end());
                const auto& gap_embedding = model.embedding(gap_tokens);
                base::Status gap_status = model.prefill(
                    gap_embedding.input_embeddings, gap, actual_kv_len);
                if (!gap_status) {
                    LOG(WARNING) << "KV gap fill failed: " << gap_status.get_err_code();
                } else {
                    if (model.get_cuda_config()) {
                        cudaStreamSynchronize(model.get_cuda_config()->stream);
                    }
                }
            }
            
            conv.update_cached_tokens(history_tokens_i32);
            
            if (config.use_prefix_cache && cache_manager) {
                cache_manager->register_prefix(history_tokens_i32,
                                               static_cast<int32_t>(history_tokens_i32.size()));
            }
            
            stats.print(!config.stream_output);
            
            stats.accumulate();
            cumulative_stats.total_prefill_tokens += stats.prefill_tokens;
            cumulative_stats.total_decode_tokens += stats.decode_steps;
            cumulative_stats.total_prefill_time_ms += stats.prefill_time_ms;
            cumulative_stats.total_decode_time_ms += stats.decode_time_ms;
            cumulative_stats.request_count++;
        }
    }
}

/**
 * @brief 运行单次推理
 * 
 * @tparam ModelType 模型类型
 */
template<typename ModelType>
void run_single_inference(
    ModelType& model,
    const InferenceConfig& config,
    PrefixCacheManager* cache_manager,
    const ModelInferConfig& model_config) {
    
    MultiTurnConversation conv;
    
    std::string user_input = "你好，请介绍一下你自己！";
    
    if (!config.stream_output) {
        LOG(INFO) << "User: " << user_input;
    }
    
    PerfStats stats;
    std::string response = generate_response(model, conv, user_input, config, stats,
                                             cache_manager, model_config);
    
    if (!config.stream_output) {
        LOG(INFO) << "\nAssistant: " << model_config.post_process(response);
    } else {
        std::cout << std::endl;
    }
    
    stats.print(!config.stream_output);
}

// ==================== 性能基准测试 ====================

/**
 * @brief 单次benchmark测试结果
 */
struct BenchmarkResult {
    int input_tokens = 0;
    int output_tokens = 0;
    double prefill_time_ms = 0.0;
    double decode_time_ms = 0.0;
    double prefill_throughput = 0.0;  // tokens/s
    double decode_throughput = 0.0;   // tokens/s
    double decode_latency_ms = 0.0;   // ms/token
};

/**
 * @brief 运行性能基准测试
 * 
 * 对不同输入token数（256, 512, 1024, 2048, 4096）进行测试，
 * 每次测试固定输出token数，收集prefill和decode性能数据。
 * 
 * @tparam ModelType 模型类型
 */
template<typename ModelType>
void run_benchmark(
    ModelType& model,
    const InferenceConfig& config,
    const ModelInferConfig& model_config) {
    
    const std::vector<int> input_lengths = {256, 512, 1024, 2048, 4096};
    const int decode_tokens = config.benchmark_decode_tokens;
    
    LOG(INFO) << "\n========================================";
    LOG(INFO) << "  Performance Benchmark (" << model_config.model_name << ")";
    LOG(INFO) << "  Decode tokens: " << decode_tokens;
    LOG(INFO) << "  FP16: " << (model.is_fp16_model() ? "yes" : "no");
    LOG(INFO) << "  AWQ: " << (model.is_awq_model() ? "yes" : "no");
    LOG(INFO) << "  CUDA Graph: " << (config.use_cuda_graph ? "enabled" : "disabled");
    LOG(INFO) << "  Fused FFN: " << (config.use_fused_ffn ? "enabled" : "disabled");
    LOG(INFO) << "========================================\n";
    
    // 生成足够长的token序列：编码一段长文本，循环填充到最大需要的长度
    std::string long_text;
    // 使用中文文本，每个汉字约1-2个token
    std::string base_text = "人工智能是计算机科学的一个分支，它企图了解智能的实质，"
        "并生产出一种新的能以人类智能相似的方式做出反应的智能机器。"
        "该领域的研究包括机器人、语言识别、图像识别、自然语言处理和专家系统等。"
        "人工智能从诞生以来，理论和技术日益成熟，应用领域也不断扩大。"
        "可以设想，未来人工智能带来的科技产品，将会是人类智慧的容器。"
        "人工智能可以对人的意识、思维的信息过程进行模拟。"
        "人工智能不是人的智能，但能像人那样思考，也可能超过人的智能。"
        "目前人工智能的主要研究方向包括深度学习、强化学习、自然语言处理、计算机视觉等。"
        "深度学习是机器学习的一个分支，它使用多层神经网络来学习数据的特征表示。"
        "强化学习是一种通过与环境交互来学习最优策略的方法。"
        "自然语言处理是让计算机理解和生成人类语言的技术。"
        "计算机视觉是让计算机理解图像和视频内容的技术。";
    
    // 重复文本直到足够长
    for (int i = 0; i < 50; ++i) {
        long_text += base_text;
    }
    
    // 编码长文本获取token序列
    auto all_tokens = model.encode(long_text);
    LOG(INFO) << "Base token pool size: " << all_tokens.size();
    
    if (static_cast<int>(all_tokens.size()) < input_lengths.back()) {
        LOG(WARNING) << "Token pool (" << all_tokens.size() 
                     << ") smaller than max input (" << input_lengths.back() 
                     << "), will cycle tokens";
    }
    
    std::vector<BenchmarkResult> results;
    
    // Warmup: 运行一次短的prefill+decode来预热GPU
    {
        LOG(INFO) << "Warming up GPU...";
        model.clear_kv_cache();
        std::vector<int32_t> warmup_tokens(all_tokens.begin(), 
            all_tokens.begin() + std::min(64, static_cast<int>(all_tokens.size())));
        const auto& warmup_emb = model.embedding(warmup_tokens);
        model.prefill(warmup_emb.input_embeddings, static_cast<int32_t>(warmup_tokens.size()), 0);
        
        // 做几步decode预热
        tensor::Tensor warmup_output = model.get_buffer(model::ModelBufferType::kForwardOutput);
        std::vector<float> warmup_logits(warmup_output.size());
        cudaMemcpy(warmup_logits.data(), warmup_output.template ptr<float>(),
                   warmup_output.size() * sizeof(float), cudaMemcpyDeviceToHost);
        int32_t warmup_next = sample_argmax(warmup_logits);
        int32_t warmup_pos = static_cast<int32_t>(warmup_tokens.size());
        for (int w = 0; w < 10; ++w) {
            std::vector<int32_t> st = {warmup_next};
            const auto& te = model.embedding(st);
            tensor::Tensor pt = model.get_buffer(model::ModelBufferType::kInputPos);
            pt.index<int32_t>(0) = warmup_pos;
            tensor::Tensor inp = model.fill_input(pt, te, false);
            model.decode(inp, warmup_pos, warmup_next);
            warmup_pos++;
        }
        if (model.get_cuda_config()) {
            cudaStreamSynchronize(model.get_cuda_config()->stream);
        }
        LOG(INFO) << "Warmup complete.\n";
    }
    
    for (int target_input : input_lengths) {
        LOG(INFO) << "--- Testing input_tokens=" << target_input 
                  << ", output_tokens=" << decode_tokens << " ---";
        
        // 清空KV cache
        model.clear_kv_cache();
        
        // 准备输入tokens：从token pool截取或循环填充到目标长度
        std::vector<int32_t> input_tokens;
        input_tokens.reserve(target_input);
        for (int i = 0; i < target_input; ++i) {
            input_tokens.push_back(all_tokens[i % all_tokens.size()]);
        }
        
        // 检查是否超过最大上下文长度
        int32_t max_seq_len = model.get_config()->seq_len_;
        if (target_input + decode_tokens > max_seq_len) {
            LOG(WARNING) << "Skipping: input(" << target_input << ") + decode(" 
                         << decode_tokens << ") > max_seq_len(" << max_seq_len << ")";
            continue;
        }
        
        BenchmarkResult result;
        result.input_tokens = target_input;
        
        // === Prefill 阶段 ===
        const auto& embedding_out = model.embedding(input_tokens);
        
        Timer prefill_timer;
        prefill_timer.start();
        
        base::Status prefill_status = model.prefill(
            embedding_out.input_embeddings, target_input, 0);
        
        if (model.get_cuda_config()) {
            cudaStreamSynchronize(model.get_cuda_config()->stream);
        }
        
        result.prefill_time_ms = prefill_timer.elapsed_ms();
        
        if (!prefill_status) {
            LOG(ERROR) << "Prefill failed for input_tokens=" << target_input;
            continue;
        }
        
        result.prefill_throughput = target_input * 1000.0 / result.prefill_time_ms;
        
        LOG(INFO) << "  Prefill: " << target_input << " tokens, " 
                  << result.prefill_time_ms << " ms, " 
                  << result.prefill_throughput << " tokens/s";
        
        // === Decode 阶段 ===
        tensor::Tensor forward_output = model.get_buffer(model::ModelBufferType::kForwardOutput);
        std::vector<float> logits_cpu(forward_output.size());
        cudaMemcpy(logits_cpu.data(), forward_output.template ptr<float>(),
                   forward_output.size() * sizeof(float), cudaMemcpyDeviceToHost);
        
        int32_t next = sample_argmax(logits_cpu);
        int32_t pos = target_input;
        int actual_decode_steps = 0;
        
        Timer decode_timer;
        decode_timer.start();
        
        for (int step = 0; step < decode_tokens; ++step) {
            std::vector<int32_t> single_token = {next};
            const auto& token_embedding = model.embedding(single_token);
            
            tensor::Tensor pos_tensor = model.get_buffer(model::ModelBufferType::kInputPos);
            pos_tensor.index<int32_t>(0) = pos;
            tensor::Tensor input = model.fill_input(pos_tensor, token_embedding, false);
            
            auto decode_status = model.decode(input, pos, next);
            if (!decode_status) {
                LOG(ERROR) << "Decode failed at step " << step;
                break;
            }
            
            pos++;
            actual_decode_steps++;
            
            // 注意：benchmark模式下不因EOS提前终止，确保测试完整的decode_tokens数
        }
        
        if (model.get_cuda_config()) {
            cudaStreamSynchronize(model.get_cuda_config()->stream);
        }
        
        result.decode_time_ms = decode_timer.elapsed_ms();
        result.output_tokens = actual_decode_steps;
        result.decode_throughput = actual_decode_steps * 1000.0 / result.decode_time_ms;
        result.decode_latency_ms = result.decode_time_ms / actual_decode_steps;
        
        LOG(INFO) << "  Decode: " << actual_decode_steps << " tokens, " 
                  << result.decode_time_ms << " ms, " 
                  << result.decode_throughput << " tokens/s, "
                  << result.decode_latency_ms << " ms/token";
        LOG(INFO) << "";
        
        results.push_back(result);
    }
    
    // === 输出汇总表格 ===
    LOG(INFO) << "\n========================================";
    LOG(INFO) << "  Benchmark Summary (" << model_config.model_name << ")";
    LOG(INFO) << "========================================";
    
    // 以CSV友好格式输出，方便后续解析
    std::cout << "BENCHMARK_CSV_START" << std::endl;
    std::cout << "model_name,input_tokens,output_tokens,"
              << "prefill_time_ms,prefill_throughput_tps,"
              << "decode_time_ms,decode_throughput_tps,decode_latency_ms" << std::endl;
    
    for (const auto& r : results) {
        char buf[512];
        snprintf(buf, sizeof(buf), "%s,%d,%d,%.2f,%.2f,%.2f,%.2f,%.2f",
                 model_config.model_name.c_str(),
                 r.input_tokens, r.output_tokens,
                 r.prefill_time_ms, r.prefill_throughput,
                 r.decode_time_ms, r.decode_throughput, r.decode_latency_ms);
        std::cout << buf << std::endl;
    }
    std::cout << "BENCHMARK_CSV_END" << std::endl;
    
    // 友好的表格输出
    LOG(INFO) << "\n+-------------+-------------+---------------+--------------+---------------+--------------+";
    LOG(INFO) << "| Input Tokens|Output Tokens| Prefill(ms)   |Prefill(tok/s)|  Decode(ms)   |Decode(tok/s) |";
    LOG(INFO) <<   "+-------------+-------------+---------------+--------------+---------------+--------------+";
    for (const auto& r : results) {
        char line[256];
        snprintf(line, sizeof(line), "| %11d | %11d | %13.2f | %12.2f | %13.2f | %12.2f |",
                 r.input_tokens, r.output_tokens,
                 r.prefill_time_ms, r.prefill_throughput,
                 r.decode_time_ms, r.decode_throughput);
        LOG(INFO) << line;
    }
    LOG(INFO) <<   "+-------------+-------------+---------------+--------------+---------------+--------------+";
    LOG(INFO) << "";
}

/**
 * @brief 通用模型推理入口
 * 
 * 封装了模型初始化、配置和运行的通用流程。
 * 
 * @tparam ModelType 模型类型
 * @param argc 命令行参数个数
 * @param argv 命令行参数
 * @param description 程序描述
 * @param model_config 模型行为配置
 * @param default_cuda_graph 是否默认启用CUDA Graph
 * @return 退出码
 */
template<typename ModelType>
int run_model_inference(
    int argc, char* argv[],
    const std::string& description,
    const ModelInferConfig& model_config,
    bool default_cuda_graph = false) {
    
    if (argc < 3) {
        print_usage(argv[0], description);
        return -1;
    }
    
    const char* checkpoint_path = argv[1];
    const char* tokenizer_path = argv[2];
    
    // 解析配置
    InferenceConfig config = InferenceConfig::parse_args(argc, argv, 3);
    
    // 处理默认 CUDA Graph 行为
    if (default_cuda_graph) {
        bool has_no_cuda_graph = false;
        for (int i = 3; i < argc; ++i) {
            if (std::string(argv[i]) == "--no-cuda-graph") {
                has_no_cuda_graph = true;
                break;
            }
        }
        if (!has_no_cuda_graph) {
            config.use_cuda_graph = true;
        }
    }
    
    // 初始化 RadixTree PrefixCache（如果启用）
    std::unique_ptr<PrefixCacheManager> cache_manager;
    if (config.use_prefix_cache) {
        cache_manager = std::make_unique<PrefixCacheManager>(config.prefix_cache_size);
        LOG(INFO) << "RadixTree PrefixCache initialized with max " 
                  << config.prefix_cache_size << " tokens";
    }
    
    // 初始化模型
    ModelType model(base::TokenizerType::kEncodeBpe, tokenizer_path,
                    checkpoint_path, false);
    auto init_status = model.init(base::DeviceType::kDeviceCUDA);
    if (!init_status) {
        LOG(FATAL) << "The model init failed, the error code is: " << init_status.get_err_code();
    }
    
    LOG(INFO) << model_config.model_name << " model loaded successfully!";
    LOG(INFO) << "Model is FP16: " << (model.is_fp16_model() ? "yes" : "no");
    
    // 配置模型选项
    model.enable_fused_ffn(config.use_fused_ffn);
    model.set_attention_type(config.attention_type);
    
    if (config.use_cuda_graph) {
        model.enable_cuda_graph(true);
    }
    
    config.print();
    
    // 运行模式选择
    if (config.benchmark_mode) {
        run_benchmark(model, config, model_config);
    } else if (config.interactive_mode) {
        run_interactive(model, config, cache_manager.get(), model_config);
    } else {
        run_single_inference(model, config, cache_manager.get(), model_config);
    }
    
    return 0;
}

}  // namespace inference

#endif  // KUIPER_DEMO_INFERENCE_COMMON_H_
