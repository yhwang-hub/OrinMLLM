# OrinMLLM 多轮对话与流式输出实现分析报告

## 目录

- [一、多轮对话的实现](#一多轮对话的实现)
  - [1.1 整体架构概览](#11-整体架构概览)
  - [1.2 对话历史的存储与管理](#12-对话历史的存储与管理)
  - [1.3 Chat Template（Jinja 模板）的应用](#13-chat-templatejinja-模板的应用)
  - [1.4 KV Cache 增量复用机制](#14-kv-cache-增量复用机制)
  - [1.5 完整流程详解](#15-完整流程详解)
  - [1.6 历史截断与上下文溢出处理](#16-历史截断与上下文溢出处理)
- [二、流式输出的实现](#二流式输出的实现)
  - [2.1 流式输出整体架构](#21-流式输出整体架构)
  - [2.2 Token 逐个生成机制](#22-token-逐个生成机制)
  - [2.3 流式输出的增量解码策略](#23-流式输出的增量解码策略)
  - [2.4 Token 解码原理（BPE → 文本）](#24-token-解码原理bpe--文本)
  - [2.5 完整流程详解](#25-完整流程详解)
  - [2.6 流式模式下的性能统计](#26-流式模式下的性能统计)
- [三、架构总结](#三架构总结)

---

## 一、多轮对话的实现

### 1.1 整体架构概览

OrinMLLM 工程中有 **两套** 多轮对话实现：

| 实现方式 | 文件 | 特点 |
|----------|------|------|
| 简易版 | `demo/chat_qwen.cpp` | 手动拼接 ChatML 格式，每轮全量 prefill，无 KV Cache 复用 |
| 正式版 | `demo/inference_common.h` | Jinja 模板渲染 + KV Cache 增量复用 + RadixTree PrefixCache |

正式版通过模板化框架 `run_model_inference<ModelType>()` 支持所有 Qwen 系列模型（Qwen2/2.5/3），并通过 `--interactive` 命令行参数启用交互式多轮对话模式。

入口程序示例（`demo/main_qwen3.cpp`）：

```cpp
int main(int argc, char* argv[]) {
    inference::ModelInferConfig model_config;
    model_config.skip_tokens = {151645};  // EOS token ID
    model_config.remove_thinking = true;  // Qwen3 支持 <think> 思考模式
    model_config.model_name = "Qwen3";
    
    return inference::run_model_inference<model::Qwen3Model>(
        argc, argv,
        "Qwen3 Model Inference with Multi-Turn Dialog and RadixTree PrefixCache",
        model_config,
        true  // 默认启用 CUDA Graph
    );
}
```

### 1.2 对话历史的存储与管理

#### 1.2.1 简易版（chat_qwen.cpp）

使用自定义 `ChatMessage` 结构体存储消息：

```cpp
// demo/chat_qwen.cpp L11-L14
struct ChatMessage {
  std::string role;     // 角色: system, user, assistant
  std::string content;  // 内容
};
```

在 `main()` 函数中用 `std::vector<ChatMessage>` 维护聊天历史：

```cpp
// demo/chat_qwen.cpp L168
std::vector<ChatMessage> chat_history;
chat_history.push_back({"system", "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."});
```

每轮对话时，手动将用户消息 push 到历史（L203），生成回复后再将 assistant 消息 push 到历史（L218）：

```cpp
// 用户消息
chat_history.push_back({"user", user_input});

// 生成回复
ChatMessage response = assistant.chat(chat_history, gen_config);

// 助手消息
chat_history.push_back(response);
```

#### 1.2.2 正式版（inference_common.h 的 MultiTurnConversation 类）

`MultiTurnConversation` 类（`demo/inference_common.h` L443-L637）是正式版的对话管理核心，内部维护以下状态：

```cpp
// demo/inference_common.h MultiTurnConversation 私有成员
std::string system_prompt_;                                  // 系统提示
std::vector<std::pair<std::string, std::string>> history_;   // (role, content) 对话历史
std::vector<int32_t> cached_tokens_;                         // 已在 KV cache 中的 tokens
int32_t current_kv_pos_ = 0;                                // 当前 KV cache 位置
```

**关键设计点**：`cached_tokens_` 存储了上一轮对话结束后完整历史对应的 token 序列，用于与下一轮的新 token 序列进行前缀比对，从而实现 KV Cache 的增量复用。

核心方法：

| 方法 | 功能 |
|------|------|
| `add_user_message(content)` | 向 `history_` 添加用户消息 |
| `add_assistant_message(content)` | 向 `history_` 添加助手消息 |
| `get_full_prompt(user_input)` | 将所有历史 + 新用户输入通过 Jinja 模板渲染为完整 prompt |
| `get_history_prompt()` | 仅渲染历史（不含新输入），用于对话结束后同步 cached_tokens |
| `compute_common_prefix_len(new_tokens)` | 逐 token 比对，计算与 cached_tokens 的公共前缀长度 |
| `update_cached_tokens(tokens)` | 更新缓存的 token 序列 |
| `append_token(token)` | decode 时逐个追加 token 到缓存 |
| `sync_cached_tokens(full_history_tokens)` | 对话结束后同步缓存 |
| `truncate_history(max_turns)` | 超过最大轮数时截断旧历史 |
| `clear()` | 清空所有状态 |

此外还有扩展版 `MultiTurnConversationWithCache`（L853-L933），在 `MultiTurnConversation` 基础上增加了 RadixTree PrefixCache 的支持，可以实现 **跨请求的前缀共享**。

### 1.3 Chat Template（Jinja 模板）的应用

#### 1.3.1 Chat Template 内容

Qwen 系列模型统一使用 ChatML 格式的 Chat Template，定义在 `demo/inference_common.h` L22-L74：

```
{%- if messages[0]['role'] == 'system' %}
    {{- '<|im_start|>system\n' + messages[0]['content'] + '<|im_end|>\n' }}
{%- else %}
    {{- '<|im_start|>system\nYou are Qwen, created by Alibaba Cloud...<|im_end|>\n' }}
{%- endif %}
{%- for message in messages %}
    {%- if (message.role == "user") or ... %}
        {{- '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>' + '\n' }}
    {%- elif ... %}
    ...
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
{%- endif %}
```

渲染后的效果（以两轮对话为例）：

```
<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
<|im_start|>user
你好<|im_end|>
<|im_start|>assistant
你好！有什么可以帮助你的吗？<|im_end|>
<|im_start|>user
介绍一下北京<|im_end|>
<|im_start|>assistant
```

#### 1.3.2 Jinja 模板引擎

Jinja 模板引擎实现在 `kuiper/include/jinja.hpp`，核心是 `jinja::Template` 类，提供 `render(context)` 和 `apply_chat_template(messages, add_generation_prompt)` 方法。

#### 1.3.3 调用链路

```
build_messages_json(system_prompt, history, user_input)   ← 构建 JSON 消息数组
    ↓
apply_chat_template(messages, true)                       ← Jinja 渲染为格式化文本
    ↓
jinja::Template(QWEN_CHAT_TEMPLATE).apply_chat_template() ← 引擎执行
```

具体代码（`demo/inference_common.h` L76-L120）：

```cpp
// 应用 chat template
inline std::string apply_chat_template(const nlohmann::json& messages, bool add_generation_prompt = true) {
    jinja::Template tpl(QWEN_CHAT_TEMPLATE);
    return tpl.apply_chat_template(messages, add_generation_prompt);
}

// 构建消息 JSON
inline nlohmann::json build_messages_json(
    const std::string& system_prompt,
    const std::vector<std::pair<std::string, std::string>>& history,
    const std::string& user_input) {
    
    nlohmann::json messages = nlohmann::json::array();
    
    messages.push_back({{"role", "system"}, {"content", system_prompt}});
    
    for (const auto& [role, content] : history) {
        messages.push_back({{"role", role}, {"content", content}});
    }
    
    if (!user_input.empty()) {
        messages.push_back({{"role", "user"}, {"content", user_input}});
    }
    
    return messages;
}
```

简易版 `chat_qwen.cpp` 则用硬编码方式拼接 ChatML（L52-L65）：

```cpp
std::string format_messages(const std::vector<ChatMessage>& messages) const {
    std::string prompt;
    for (const auto& message : messages) {
        prompt += "<|im_start|>" + message.role + "\n";
        prompt += message.content + "\n";
        prompt += "<|im_end|>\n";
    }
    prompt += "<|im_start|>assistant\n";
    return prompt;
}
```

### 1.4 KV Cache 增量复用机制

这是正式版多轮对话的核心优化。其基本思想是：**每轮对话时，重新构建完整的 prompt（包含所有历史），但通过与上一轮缓存的 token 序列进行前缀比对，只对新增 token 执行 prefill，复用已在 KV Cache 中的旧 token 的计算结果**。

#### 1.4.1 前缀比对

`compute_common_prefix_len()` 方法（`demo/inference_common.h` L555-L566）：

```cpp
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
```

#### 1.4.2 增量 Prefill

在 `generate_response()` 中（L1127-L1157），根据前缀比对结果选择增量或全量 prefill：

```cpp
if (stats.prefill_tokens > 0) {
    if (start_pos > 0) {
        // 增量 prefill: 只对新 token (tokens[start_pos:]) 计算 embedding
        std::vector<int> new_tokens(tokens.begin() + start_pos, tokens.end());
        const auto& embedding_out = model.embedding(new_tokens);
        base::Status status = model.prefill(embedding_out.input_embeddings, 
                                            stats.prefill_tokens, start_pos);
    } else {
        // 全量 prefill: 计算所有 token 的 embedding
        const auto& embedding_out = model.embedding(tokens);
        base::Status status = model.prefill(embedding_out.input_embeddings, total_len, 0);
    }
}
```

**增量 prefill 的关键参数**：`start_pos` 告诉模型 attention 层从哪个位置开始写入新的 KV 数据，前面的位置已经有上一轮的 KV 缓存。

#### 1.4.3 KV Gap 补齐

对话结束后还有一个重要步骤——**KV Gap 补齐**（L1316-L1337）。这是因为：

1. 模型 decode 时，最后输出的 EOS token（如 `<|im_end|>`）不会被写入 KV cache（检测到 EOS 后直接 break，该 token 从未作为输入经过 decode）
2. 重新 tokenize 的完整历史中包含 `<|im_end|>\n` 等分隔符 token
3. 这些 token 在 KV cache 中没有对应的计算结果

如果不补齐，下一轮对话的 prefix cache 会声称复用这些位置，导致 attention 计算使用未初始化的 KV 数据。

```cpp
int32_t actual_kv_len = stats.prompt_len + stats.decode_steps;
int32_t retokenized_len = static_cast<int32_t>(history_tokens_i32.size());

if (retokenized_len > actual_kv_len) {
    int32_t gap = retokenized_len - actual_kv_len;
    std::vector<int> gap_tokens(history_tokens_i32.begin() + actual_kv_len,
                                history_tokens_i32.end());
    const auto& gap_embedding = model.embedding(gap_tokens);
    base::Status gap_status = model.prefill(gap_embedding.input_embeddings, gap, actual_kv_len);
}
```

### 1.5 完整流程详解

以交互式模式 `run_interactive()`（L1270-L1356）为例，一轮完整的多轮对话流程如下：

```
┌─────────────────────────────────────────────────────────────────┐
│                    多轮对话完整流程                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. 用户输入                                                     │
│     └─ read_user_input() → "介绍一下北京"                        │
│                                                                 │
│  2. 命令检查                                                     │
│     └─ process_command_with_cache()                              │
│        若为 /clear、/stats、/cache、/quit 等命令则直接处理        │
│                                                                 │
│  3. 历史截断检查                                                 │
│     └─ conv.truncate_history(max_history_turns)                  │
│        超过最大轮数时删除旧消息，清空 KV Cache                    │
│                                                                 │
│  4. 生成回复 generate_response()                                 │
│     │                                                            │
│     ├─ 4a. 构建完整 prompt                                       │
│     │   conv.get_full_prompt(user_input)                         │
│     │   → build_messages_json() → apply_chat_template()          │
│     │   → "<|im_start|>system\n...<|im_end|>\n                   │
│     │      <|im_start|>user\n你好<|im_end|>\n                    │
│     │      <|im_start|>assistant\n..."                           │
│     │                                                            │
│     ├─ 4b. Tokenize                                              │
│     │   model.encode(full_prompt) → [token_id_1, ..., token_id_N]│
│     │                                                            │
│     ├─ 4c. 计算 KV Cache 可复用前缀长度                          │
│     │   方式1: conv.compute_common_prefix_len(tokens) → 线性匹配  │
│     │   方式2: cache_manager->match(tokens) → RadixTree 匹配      │
│     │   得到 start_pos（可复用的 token 数）                       │
│     │                                                            │
│     ├─ 4d. Embedding + 增量 Prefill                              │
│     │   model.embedding(tokens[start_pos:]) → 新 token 的向量    │
│     │   model.prefill(embeddings, new_len, start_pos)            │
│     │   → Transformer 前向：RMSNorm → QKV+RoPE → Attention → FFN │
│     │                                                            │
│     ├─ 4e. 采样首个 token                                        │
│     │   sample_argmax(logits) → next_token                       │
│     │                                                            │
│     ├─ 4f. Decode 循环（自回归生成）                              │
│     │   while (decode_steps < max_tokens):                       │
│     │     if is_sentence_ending(next): break                     │
│     │     model.embedding({next}) → 单 token embedding           │
│     │     model.decode(input, pos, next) → 前向传播 + argmax     │
│     │     [流式输出] printf(增量文本)                             │
│     │     pos++, decode_steps++                                  │
│     │                                                            │
│     └─ 4g. 返回生成的完整回复文本                                │
│                                                                 │
│  5. 后处理 & 更新历史                                            │
│     ├─ model_config.post_process(response)                       │
│     │   → Qwen3: 移除 <think>...</think> 思考内容                │
│     ├─ conv.add_user_message(user_input)                         │
│     ├─ conv.add_assistant_message(response)                      │
│     │                                                            │
│  6. 同步 KV Cache 状态                                           │
│     ├─ 重新 tokenize 完整历史: model.encode(get_history_prompt())│
│     ├─ KV Gap 补齐: prefill 缺失的分隔符 token                   │
│     ├─ conv.update_cached_tokens(history_tokens)                 │
│     └─ [可选] cache_manager->register_prefix(tokens)             │
│                                                                 │
│  7. 打印性能统计                                                 │
│     stats.print() → Prefill/Decode 吞吐量, KV复用率 等           │
│                                                                 │
│  8. 回到步骤 1, 等待下一轮用户输入                               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### 关键代码走读（run_interactive）

```cpp
// demo/inference_common.h L1270-L1356
template<typename ModelType>
void run_interactive(ModelType& model, const InferenceConfig& config,
                     PrefixCacheManager* cache_manager, const ModelInferConfig& model_config) {
    
    MultiTurnConversation conv;           // 对话管理器
    PerfStats cumulative_stats;           // 累计统计
    
    while (true) {
        // 1. 读取用户输入
        std::string user_input = read_user_input();
        
        // 2. 命令处理
        if (process_command_with_cache(user_input, conv, cumulative_stats, 
                                       clear_kv_cache_fn, cache_manager)) {
            continue;
        }
        
        // 3. 历史截断检查
        if (conv.truncate_history(config.max_history_turns)) {
            model.clear_kv_cache();
        }
        
        // 4. 生成回复（核心函数）
        PerfStats stats;
        std::string response = generate_response(model, conv, user_input, config, 
                                                 stats, cache_manager, model_config);
        
        // 5. 显示回复 & 更新历史
        if (!response.empty()) {
            std::string display_response = model_config.post_process(response);
            if (!config.stream_output) {
                LOG(INFO) << "\nAssistant: " << display_response;
            }
            
            // 存储完整回复（包含 thinking 内容，以保持 KV cache 一致）
            conv.add_user_message(user_input);
            conv.add_assistant_message(response);
            
            // 6. 同步 KV cache 状态
            std::string history_prompt = conv.get_history_prompt();
            auto history_tokens = model.encode(history_prompt);
            
            // KV gap 补齐
            int32_t actual_kv_len = stats.prompt_len + stats.decode_steps;
            int32_t retokenized_len = history_tokens.size();
            if (retokenized_len > actual_kv_len) {
                // 补齐分隔符 token 对应的 KV cache
                ...
            }
            
            conv.update_cached_tokens(history_tokens_i32);
            
            // 7. 打印性能统计
            stats.print(!config.stream_output);
        }
    }
}
```

### 1.6 历史截断与上下文溢出处理

#### 1.6.1 历史截断

当对话轮数超过 `max_history_turns`（默认 10 轮）时，`truncate_history()` 会删除最旧的消息（L603-L614）：

```cpp
bool truncate_history(size_t max_turns) {
    size_t max_messages = max_turns * 2;  // 每轮包含 user 和 assistant
    if (history_.size() > max_messages) {
        history_.erase(history_.begin(), history_.end() - max_messages);
        cached_tokens_.clear();
        current_kv_pos_ = 0;
        return true;  // 发生截断，KV cache 需要重建
    }
    return false;
}
```

截断后 KV Cache 完全失效，需要从头重新 prefill 所有历史。

#### 1.6.2 上下文溢出

在 `generate_response()` 中会检查总 token 数是否超过模型的 `max_seq_len`（L1085-L1092）：

```cpp
if (total_len + config.max_tokens > max_seq_len) {
    LOG(WARNING) << "Context length exceeds max_seq_len...";
    return "[Context length exceeded. Please use /clear to start a new conversation.]";
}
```

---

## 二、流式输出的实现

### 2.1 流式输出整体架构

流式输出通过 `InferenceConfig::stream_output` 标志控制（L130），由命令行 `--stream` 参数启用：

```cpp
struct InferenceConfig {
  bool stream_output = false;  // 是否流式输出
  ...
};
```

流式输出的核心思想是：**每生成一个 token，立即将其解码为文本并打印到终端，而不是等所有 token 生成完毕再一次性输出**。

### 2.2 Token 逐个生成机制

Token 生成发生在 `generate_response()` 的 decode 循环中（L1202-L1242）：

```cpp
while (decode_steps < config.max_tokens) {
    // 终止条件：遇到 EOS token
    if (model.is_sentence_ending(next)) {
        break;
    }
    
    // 1. 当前 token 转 embedding
    std::vector<int32_t> single_token = {next};
    const auto& token_embedding = model.embedding(single_token);
    
    // 2. 设置位置信息
    tensor::Tensor pos_tensor = model.get_buffer(model::ModelBufferType::kInputPos);
    pos_tensor.index<int32_t>(0) = pos;
    tensor::Tensor input = model.fill_input(pos_tensor, token_embedding, false);
    
    // 3. 执行一步 decode（Transformer 前向传播 + argmax 采样）
    auto decode_status = model.decode(input, pos, next);
    
    // 4. 追加到对话管理器缓存
    conv.append_token(next);
    
    pos++;
    decode_steps++;
}
```

每步 `model.decode()` 内部执行完整的 Transformer 前向传播：
- 遍历所有 Transformer 层：RMSNorm → QKV 投影 + RoPE → Attention → FFN
- 最后 `cls_logits()` 计算词表维度的 logits
- `post_processing()` 执行 argmax 采样得到 `next` token

终止条件通过 `model.is_sentence_ending(next)` 判断（`kuiper/source/model/model.cpp` L333-L335），最终调用 tokenizer 的 EOS 检测：

```cpp
// kuiper/source/op/encode.cpp L136-L141
bool BpeEncodeLayer::is_sentence_ending(int32_t token_id) const {
  if (token_id == stop_token1_ || token_id == stop_token2_) {  
    return true;        // stop_token1_ = <|im_end|> (151645)
  }                     // stop_token2_ = <|endoftext|>
  return false;
}
```

### 2.3 流式输出的增量解码策略

流式输出的核心实现在 decode 循环内（L1224-L1237）：

```cpp
if (!model_config.should_skip(next)) {
    generated_tokens.push_back(next);
    if (config.stream_output) {
        // 增量解码：将所有已生成 token 一起解码
        std::string decoded = static_cast<model::Model&>(model).decode(generated_tokens);
        // 只打印新增部分
        std::string new_text = decoded.substr(prev_decoded_text.length());
        if (!new_text.empty()) {
            printf("%s", new_text.c_str());
            fflush(stdout);   // 立即刷新缓冲区，确保用户实时看到输出
        }
        prev_decoded_text = decoded;
    }
}
```

**重要设计决策**：这里 **不是** 简单地解码单个 token 然后打印，而是 **每次都解码全部已生成的 tokens**，然后用 `substr` 提取相对于上次解码结果的增量文本。

这样做的原因是 BPE tokenizer 的特性：
- UTF-8 多字节字符（如中文）可能被拆分成多个 token
- 单独解码一个 token 可能产生乱码或不完整的字符
- 只有将多个 token 组合解码才能得到正确的文字

例如，一个中文字"你"可能被编码为 2 个 BPE token，单独解码第一个 token 会得到不完整的 UTF-8 字节。通过全量解码 + 增量截取的方式，确保输出始终是合法的 UTF-8 文本。

首个 token 的流式输出在 prefill 完成后立即处理（L1186-L1197），逻辑类似：

```cpp
// 首个 token 处理
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
```

### 2.4 Token 解码原理（BPE → 文本）

Token 到文本的解码通过 `Model::decode()` → `BpeEncodeLayer::decode()` 完成。

对于 Qwen 系列模型，使用 `QwenEncodeLayer::decode()`（`kuiper/source/op/encode.cpp` L200-L204）：

```cpp
std::string QwenEncodeLayer::decode(const std::vector<int32_t>& token_ids) const {
  CHECK(this->tiktoken_ != nullptr);
  // Qwen tokenizer 不需要 Ġ→空格 的替换
  return tiktoken_->decode(token_ids);
}
```

通用 BPE 解码（`kuiper/source/op/encode.cpp` L127-L133）：

```cpp
std::string BpeEncodeLayer::decode(const std::vector<int32_t>& token_ids) const {
  CHECK(this->tiktoken_ != nullptr);
  auto s = tiktoken_->decode(token_ids);
  std::map<std::string, std::string> reverse_replacements;
  reverse_replacements["Ġ"] = " ";   // BPE 特殊字符替换回空格
  const std::string& sentence = absl::StrReplaceAll(s, reverse_replacements);
  return sentence;
}
```

特殊 token 跳过机制通过 `ModelInferConfig::should_skip()` 实现（L1040-L1045），过滤 EOS/BOS 等不应输出的 token：

```cpp
bool should_skip(int32_t token) const {
    for (auto t : skip_tokens) {
        if (t == token) return true;
    }
    return false;
}
```

### 2.5 完整流程详解

```
┌──────────────────────────────────────────────────────────────┐
│                    流式输出完整流程                            │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  Prefill 阶段                                                │
│  ═══════════                                                 │
│  model.prefill(embeddings) → 前向传播所有 prompt tokens       │
│      ↓                                                       │
│  sample_argmax(logits) → 首个 token (next)                   │
│      ↓                                                       │
│  [流式] generated_tokens = [next]                            │
│         decoded = model.decode([next]) → "你"                │
│         printf("你"), fflush(stdout) → 用户立即看到 "你"      │
│                                                              │
│  Decode 循环                                                 │
│  ═══════════                                                 │
│  Step 1:                                                     │
│    model.embedding({next}) → 单 token embedding              │
│    model.decode(input, pos, next) → Transformer → argmax     │
│    next = 新 token                                           │
│    [流式] generated_tokens = [t1, t2]                        │
│           decoded = model.decode([t1, t2]) → "你好"           │
│           new_text = "你好"[len("你"):] → "好"               │
│           printf("好"), fflush(stdout)                        │
│                                                              │
│  Step 2:                                                     │
│    model.decode(...) → next                                   │
│    [流式] generated_tokens = [t1, t2, t3]                     │
│           decoded = "你好！" → new_text = "！"                │
│           printf("！"), fflush(stdout)                        │
│                                                              │
│  ...继续直到 is_sentence_ending(next) 或达到 max_tokens...    │
│                                                              │
│  结束处理                                                    │
│  ═══════════                                                 │
│  [流式模式]  printf("\n")    ← 仅打印换行                    │
│  [非流式]    LOG(INFO) << "Assistant: " << response           │
│                                                              │
│  性能统计                                                    │
│  [流式模式]  使用 stderr 输出紧凑格式，避免干扰 stdout        │
│  [非流式]    使用 LOG(INFO) 输出详细格式                      │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### 2.6 流式模式下的性能统计

流式模式下，性能统计使用紧凑格式输出到 `stderr`（而非 `stdout`），避免干扰流式文本输出（L419-L427）：

```cpp
void print(bool verbose = true) const {
    if (verbose) {       // 非流式：详细格式输出到 LOG(INFO)
        LOG(INFO) << "\n=== Performance Statistics ===";
        LOG(INFO) << "Prefill: " << prefill_tokens << " tokens, " 
                  << prefill_throughput() << " tokens/s";
        LOG(INFO) << "Decode: " << decode_steps << " tokens, " 
                  << decode_throughput() << " tokens/s";
    } else {             // 流式：紧凑格式输出到 stderr
        std::cerr << "\n[Prefill: " << prefill_tokens << " tokens, " 
                  << prefill_throughput() << " tokens/s"
                  << " | Decode: " << decode_steps << " tokens, " 
                  << decode_throughput() << " tokens/s]" << std::endl;
    }
}
```

在 `run_interactive()` 中通过 `!config.stream_output` 控制调用哪种格式：

```cpp
stats.print(!config.stream_output);  // stream 模式用紧凑格式
```

---

## 三、架构总结

### 组件关系图

```
┌──────────────────────────────────────────────────────────────────┐
│                        入口程序层                                │
│  main_qwen.cpp / main_qwen3.cpp / main_qwen3_vl.cpp            │
│  → 配置 ModelInferConfig，调用 run_model_inference<ModelType>()  │
└──────────────────────────┬───────────────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────────────┐
│                      推理框架层 (inference_common.h)              │
│                                                                  │
│  ┌─────────────────┐  ┌──────────────────┐  ┌────────────────┐  │
│  │ InferenceConfig  │  │ MultiTurn        │  │ PrefixCache    │  │
│  │ ─────────────── │  │ Conversation     │  │ Manager        │  │
│  │ stream_output   │  │ ──────────────── │  │ ────────────── │  │
│  │ use_cuda_graph  │  │ history_         │  │ RadixTree      │  │
│  │ max_tokens      │  │ cached_tokens_   │  │ match/register │  │
│  │ interactive     │  │ kv_position      │  │ LRU eviction   │  │
│  └─────────────────┘  └──────────────────┘  └────────────────┘  │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │       generate_response<ModelType>() ← 核心生成函数         │ │
│  │  1. get_full_prompt → Jinja渲染                             │ │
│  │  2. encode → tokenize                                       │ │
│  │  3. compute_common_prefix → KV复用                          │ │
│  │  4. embedding + prefill → 增量前向                          │ │
│  │  5. decode loop → 自回归 + 流式输出                         │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  ┌─────────────────┐  ┌──────────────────┐                      │
│  │ run_interactive  │  │ run_single       │                      │
│  │ (多轮对话循环)  │  │ (单次推理)       │                      │
│  └─────────────────┘  └──────────────────┘                      │
└──────────────────────────┬───────────────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────────────┐
│                        模型层                                    │
│  kuiper/source/model/qwen2.cpp | qwen3.cpp                     │
│  ─────────────────────────────────                              │
│  prefill() → 批量前向传播（所有 prompt tokens）                  │
│  decode()  → 单步前向传播（单个 token + KV Cache）               │
│  embedding() → token ID → 向量                                  │
│  clear_kv_cache() → 重置 KV Cache                               │
└──────────────────────────┬───────────────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────────────┐
│                      Tokenizer 层                                │
│  kuiper/source/op/encode.cpp                                    │
│  ────────────────────────                                       │
│  QwenEncodeLayer::encode() → 文本 → token IDs (BPE)             │
│  QwenEncodeLayer::decode() → token IDs → 文本 (tiktoken)        │
│  is_sentence_ending() → EOS 检测 (<|im_end|>, <|endoftext|>)   │
└──────────────────────────┬───────────────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────────────┐
│                    Chat Template 层                               │
│  kuiper/include/jinja.hpp + inference_common.h                  │
│  ────────────────────────────────                                │
│  QWEN_CHAT_TEMPLATE → ChatML 格式模板                           │
│  jinja::Template::apply_chat_template() → 渲染消息为格式化文本   │
│  build_messages_json() → 构建 JSON 消息数组                     │
└─────────────────────────────────────────────────────────────────┘
```

### 关键设计亮点

1. **KV Cache 增量复用**：通过 token 序列前缀比对，多轮对话中只对新增 token 执行 prefill，大幅减少计算量。

2. **KV Gap 补齐机制**：解决了 EOS token 不进入 KV Cache 导致的一致性问题，保证后续轮次的前缀复用正确性。

3. **增量解码流式输出**：每次全量解码所有已生成 token 再取增量，完美处理 BPE 编码中跨 token 的 UTF-8 字符边界问题。

4. **模板化设计**：通过 `run_model_inference<ModelType>()` 和 `ModelInferConfig` 实现一套代码支持多种模型（Qwen2/2.5/3），仅需配置差异参数。

5. **RadixTree PrefixCache**：可选的树结构前缀缓存，支持更高效的跨请求前缀匹配和 LRU 淘汰。
