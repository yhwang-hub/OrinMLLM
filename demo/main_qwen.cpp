/**
 * @file main_qwen.cpp
 * @brief Qwen2/Qwen2.5 模型推理Demo（支持多轮对话和RadixTree PrefixCache）
 * 
 * 运行示例:
 *   ./demo/qwen_infer model.bin tokenizer.json -i --stream --max-tokens 512 --prefix-cache
 */

#include "model/qwen2.h"
#include "inference_common.h"

int main(int argc, char* argv[]) {
    inference::ModelInferConfig model_config;
    model_config.skip_tokens = {151645, 151644};  // EOS + BOS
    model_config.remove_thinking = false;
    model_config.model_name = "Qwen2/2.5";
    
    return inference::run_model_inference<model::Qwen2Model>(
        argc, argv,
        "Qwen2/Qwen2.5 Model Inference with Multi-Turn Dialog and RadixTree PrefixCache",
        model_config,
        true  // 不默认启用CUDA Graph
    );
}
