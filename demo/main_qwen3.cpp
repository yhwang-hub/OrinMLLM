/**
 * @file main_qwen3.cpp
 * @brief Qwen3 模型推理Demo（支持多轮对话和RadixTree PrefixCache）
 * 
 * 运行示例:
 *   ./demo/qwen3_infer model.bin tokenizer.json -i --stream --max-tokens 1024 --prefix-cache
 */

#include "model/qwen3.h"
#include "inference_common.h"

int main(int argc, char* argv[]) {
    inference::ModelInferConfig model_config;
    model_config.skip_tokens = {151645};  // EOS only
    model_config.remove_thinking = true;  // Qwen3 支持 <think> 思考模式
    model_config.model_name = "Qwen3";
    
    return inference::run_model_inference<model::Qwen3Model>(
        argc, argv,
        "Qwen3 Model Inference with Multi-Turn Dialog and RadixTree PrefixCache",
        model_config,
        true  // Qwen3 默认启用 CUDA Graph
    );
}
