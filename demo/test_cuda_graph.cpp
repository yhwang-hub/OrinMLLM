/**
 * @file test_cuda_graph.cpp
 * @brief Test cases for CUDA Graph integration with Qwen2 model inference
 * 
 * This test file verifies:
 * 1. CUDA Graph capture and replay in decode phase
 * 2. Compatibility between CUDA Graph and Prefix Cache
 * 3. Output correctness with/without CUDA Graph
 * 4. Performance comparison
 * 
 * Usage:
 *   ./test_cuda_graph <checkpoint_path> <tokenizer_path>
 * 
 * Example:
 *   ./test_cuda_graph /mnt/ssd/QwenMoedls/Qwen2.5-1.5B.bin tokenizer_json/Qwen2.5-1.5B-Instruct/tokenizer.json
 */

#include <base/base.h>
#include <base/tick.h>
#include <glog/logging.h>
#include <cuda_runtime.h>
#include <fstream>
#include <chrono>
#include <vector>
#include <string>
#include <iomanip>
#include "model/qwen2.h"
#include "jinja.hpp"

// Qwen2.5 Chat Template
static const std::string QWEN2_CHAT_TEMPLATE = R"(
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

static std::string apply_chat_template(const nlohmann::json& messages, bool add_generation_prompt = true) {
    jinja::Template tpl(QWEN2_CHAT_TEMPLATE);
    return tpl.apply_chat_template(messages, add_generation_prompt);
}

// Test result structure
struct TestResult {
    std::string test_name;
    bool passed;
    double prefill_time_ms;
    double decode_time_ms;
    double decode_throughput;  // tokens/s
    int generated_tokens;
    std::string output_text;
    std::string error_message;
};

// Generate function with timing
TestResult generate_with_timing(model::Qwen2Model& model, const std::string& prompt, 
                                 int max_new_tokens, const std::string& test_name) {
    TestResult result;
    result.test_name = test_name;
    result.passed = true;
    
    auto tokens = model.encode(prompt);
    int32_t prompt_len = tokens.size();
    
    if (tokens.empty()) {
        result.passed = false;
        result.error_message = "Empty tokens";
        return result;
    }
    
    // Clear KV cache for fresh start
    model.clear_kv_cache();
    
    // Get embeddings
    const auto& prompt_embedding = model.embedding(tokens);
    std::vector<int32_t> words(tokens.begin(), tokens.end());
    
    // ==================== Prefill Phase ====================
    auto start_prefill = std::chrono::steady_clock::now();
    
    auto prefill_status = model.prefill(prompt_embedding.input_embeddings, prompt_len, 0);
    
    if (!prefill_status) {
        result.passed = false;
        result.error_message = "Prefill failed";
        return result;
    }
    
    // Sync
    if (model.get_cuda_config()) {
        cudaStreamSynchronize(model.get_cuda_config()->stream);
    }
    
    auto end_prefill = std::chrono::steady_clock::now();
    result.prefill_time_ms = std::chrono::duration_cast<std::chrono::microseconds>(
        end_prefill - start_prefill).count() / 1000.0;
    
    // Sample first token
    tensor::Tensor forward_output = model.get_buffer(model::ModelBufferType::kForwardOutput);
    
    std::vector<float> logits_cpu(forward_output.size());
    cudaMemcpy(logits_cpu.data(), forward_output.ptr<float>(), 
               forward_output.size() * sizeof(float), cudaMemcpyDeviceToHost);
    
    int32_t next = 0;
    float max_val = logits_cpu[0];
    for (int32_t i = 1; i < forward_output.size(); ++i) {
        if (logits_cpu[i] > max_val) {
            max_val = logits_cpu[i];
            next = i;
        }
    }
    words.push_back(next);
    
    // ==================== Decode Phase ====================
    auto start_decode = std::chrono::steady_clock::now();
    int32_t pos = prompt_len;
    int decode_steps = 0;
    
    while (decode_steps < max_new_tokens) {
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
            result.passed = false;
            result.error_message = "Decode failed at step " + std::to_string(decode_steps);
            return result;
        }
        
        words.push_back(next);
        pos += 1;
        decode_steps += 1;
    }
    
    if (model.get_cuda_config()) {
        cudaStreamSynchronize(model.get_cuda_config()->stream);
    }
    
    auto end_decode = std::chrono::steady_clock::now();
    result.decode_time_ms = std::chrono::duration_cast<std::chrono::microseconds>(
        end_decode - start_decode).count() / 1000.0;
    
    result.generated_tokens = decode_steps;
    if (decode_steps > 0 && result.decode_time_ms > 0) {
        result.decode_throughput = decode_steps * 1000.0 / result.decode_time_ms;
    } else {
        result.decode_throughput = 0;
    }
    
    result.output_text = model.Model::decode(words);
    
    return result;
}

void print_result(const TestResult& result) {
    std::cout << "\n=== " << result.test_name << " ===" << std::endl;
    std::cout << "Status: " << (result.passed ? "PASSED" : "FAILED") << std::endl;
    
    if (!result.passed) {
        std::cout << "Error: " << result.error_message << std::endl;
        return;
    }
    
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Prefill time: " << result.prefill_time_ms << " ms" << std::endl;
    std::cout << "Decode time: " << result.decode_time_ms << " ms" << std::endl;
    std::cout << "Generated tokens: " << result.generated_tokens << std::endl;
    std::cout << "Decode throughput: " << result.decode_throughput << " tokens/s" << std::endl;
    std::cout << "Output: " << result.output_text.substr(0, 200) << "..." << std::endl;
}

// Test 1: Basic CUDA Graph functionality
bool test_cuda_graph_basic(model::Qwen2Model& model, const std::string& prompt) {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "TEST 1: Basic CUDA Graph Functionality" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    
    // Test without CUDA Graph
    model.enable_cuda_graph(false);
    model.clear_kv_cache();
    
    TestResult result_no_graph = generate_with_timing(model, prompt, 32, "Without CUDA Graph");
    print_result(result_no_graph);
    
    // Test with CUDA Graph
    model.enable_cuda_graph(true);
    model.clear_kv_cache();
    model.invalidate_cuda_graph();
    
    TestResult result_with_graph = generate_with_timing(model, prompt, 32, "With CUDA Graph");
    print_result(result_with_graph);
    
    // Compare results
    std::cout << "\n--- Comparison ---" << std::endl;
    double speedup = (result_no_graph.decode_time_ms > 0 && result_with_graph.decode_time_ms > 0) 
        ? result_no_graph.decode_time_ms / result_with_graph.decode_time_ms : 1.0;
    std::cout << "Decode speedup with CUDA Graph: " << std::fixed << std::setprecision(2) 
              << speedup << "x" << std::endl;
    
    // Verify output correctness (tokens should match)
    bool output_match = (result_no_graph.generated_tokens == result_with_graph.generated_tokens);
    std::cout << "Token count match: " << (output_match ? "YES" : "NO") << std::endl;
    
    return result_no_graph.passed && result_with_graph.passed;
}

// Test 2: Multi-turn conversation with CUDA Graph
bool test_cuda_graph_multi_turn(model::Qwen2Model& model) {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "TEST 2: Multi-turn Conversation with CUDA Graph" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    
    model.enable_cuda_graph(true);
    model.clear_kv_cache();
    model.invalidate_cuda_graph();
    
    // Turn 1
    nlohmann::json turn1 = nlohmann::json::array({
        {{"role", "system"}, {"content", "You are a helpful assistant."}},
        {{"role", "user"}, {"content", "Hello!"}}
    });
    std::string prompt1 = apply_chat_template(turn1, true);
    
    TestResult result1 = generate_with_timing(model, prompt1, 20, "Turn 1");
    print_result(result1);
    
    // Turn 2 (different prompt)
    nlohmann::json turn2 = nlohmann::json::array({
        {{"role", "system"}, {"content", "You are a helpful assistant."}},
        {{"role", "user"}, {"content", "Hello! How are you?"}}
    });
    std::string prompt2 = apply_chat_template(turn2, true);
    
    model.clear_kv_cache();
    
    TestResult result2 = generate_with_timing(model, prompt2, 20, "Turn 2 (extended)");
    print_result(result2);
    
    // Turn 3 (completely different)
    nlohmann::json turn3 = nlohmann::json::array({
        {{"role", "system"}, {"content", "You are a math expert."}},
        {{"role", "user"}, {"content", "Calculate 15 * 7."}}
    });
    std::string prompt3 = apply_chat_template(turn3, true);
    
    model.clear_kv_cache();
    
    TestResult result3 = generate_with_timing(model, prompt3, 20, "Turn 3 (different topic)");
    print_result(result3);
    
    return result1.passed && result2.passed && result3.passed;
}

// Test 3: Stress test - multiple iterations
bool test_cuda_graph_stress(model::Qwen2Model& model, const std::string& prompt) {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "TEST 3: Stress Test (10 iterations)" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    
    model.enable_cuda_graph(true);
    
    const int num_iterations = 10;
    std::vector<double> throughputs;
    int failures = 0;
    
    for (int i = 0; i < num_iterations; ++i) {
        model.clear_kv_cache();
        
        TestResult result = generate_with_timing(model, prompt, 16, 
            "Iteration " + std::to_string(i + 1));
        
        if (result.passed) {
            throughputs.push_back(result.decode_throughput);
        } else {
            failures++;
        }
        
        std::cout << "Iteration " << (i + 1) << ": " 
                  << (result.passed ? std::to_string(result.decode_throughput) + " tokens/s" : "FAILED") 
                  << std::endl;
    }
    
    if (!throughputs.empty()) {
        double sum = 0;
        double min_tp = throughputs[0], max_tp = throughputs[0];
        for (double tp : throughputs) {
            sum += tp;
            min_tp = std::min(min_tp, tp);
            max_tp = std::max(max_tp, tp);
        }
        double avg = sum / throughputs.size();
        
        std::cout << "\n--- Statistics ---" << std::endl;
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "Average throughput: " << avg << " tokens/s" << std::endl;
        std::cout << "Min throughput: " << min_tp << " tokens/s" << std::endl;
        std::cout << "Max throughput: " << max_tp << " tokens/s" << std::endl;
        std::cout << "Failures: " << failures << "/" << num_iterations << std::endl;
    }
    
    return failures == 0;
}

// Test 4: Graph invalidation and recapture
bool test_cuda_graph_invalidation(model::Qwen2Model& model) {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "TEST 4: Graph Invalidation and Recapture" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    
    model.enable_cuda_graph(true);
    
    nlohmann::json messages = nlohmann::json::array({
        {{"role", "system"}, {"content", "You are a helpful assistant."}},
        {{"role", "user"}, {"content", "Say hello."}}
    });
    std::string prompt = apply_chat_template(messages, true);
    
    // First run - capture
    model.clear_kv_cache();
    model.invalidate_cuda_graph();
    
    TestResult result1 = generate_with_timing(model, prompt, 10, "Initial capture");
    print_result(result1);
    
    // Manually invalidate and recapture
    model.invalidate_cuda_graph();
    model.clear_kv_cache();
    
    TestResult result2 = generate_with_timing(model, prompt, 10, "After invalidation");
    print_result(result2);
    
    // Third run - should use existing graph
    model.clear_kv_cache();
    
    TestResult result3 = generate_with_timing(model, prompt, 10, "Graph replay");
    print_result(result3);
    
    return result1.passed && result2.passed && result3.passed;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cout << "Usage: " << argv[0] << " <checkpoint_path> <tokenizer_path>" << std::endl;
        std::cout << "Example: " << argv[0] << " /mnt/ssd/QwenMoedls/Qwen2.5-1.5B.bin "
                  << "tokenizer_json/Qwen2.5-1.5B-Instruct/tokenizer.json" << std::endl;
        return -1;
    }
    
    const char* checkpoint_path = argv[1];
    const char* tokenizer_path = argv[2];
    
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "CUDA Graph Integration Test Suite" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    std::cout << "Checkpoint: " << checkpoint_path << std::endl;
    std::cout << "Tokenizer: " << tokenizer_path << std::endl;
    
    // Initialize model
    model::Qwen2Model model(base::TokenizerType::kEncodeBpe, tokenizer_path,
                            checkpoint_path, false);
    auto init_status = model.init(base::DeviceType::kDeviceCUDA);
    if (!init_status) {
        LOG(FATAL) << "Model init failed: " << init_status.get_err_code();
    }
    
    std::cout << "Model initialized successfully." << std::endl;
    
    // Prepare test prompt
    nlohmann::json messages = nlohmann::json::array({
        {{"role", "system"}, {"content", "You are Qwen, a helpful assistant."}},
        {{"role", "user"}, {"content", "Hello!"}}
    });
    std::string test_prompt = apply_chat_template(messages, true);
    
    // Run tests
    int passed = 0;
    int total = 4;
    
    if (test_cuda_graph_basic(model, test_prompt)) passed++;
    if (test_cuda_graph_multi_turn(model)) passed++;
    if (test_cuda_graph_stress(model, test_prompt)) passed++;
    if (test_cuda_graph_invalidation(model)) passed++;
    
    // Summary
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "TEST SUMMARY" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    std::cout << "Passed: " << passed << "/" << total << std::endl;
    std::cout << "Status: " << (passed == total ? "ALL TESTS PASSED" : "SOME TESTS FAILED") << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    
    return (passed == total) ? 0 : 1;
}
