/**
 * @file main_qwen3.cpp
 * @brief Qwen3 模型推理Demo（支持多轮对话和RadixTree PrefixCache）
 * 
 * 支持 FP16、AWQ INT4 和 SmoothQuant INT8 三种模型格式，自动检测模型类型。
 * 
 * 运行示例:
 *   ./demo/qwen3_infer model-fp16.bin tokenizer.json -i --stream --max-tokens 1024 --prefix-cache
 *   ./demo/qwen3_infer model-awq.bin  tokenizer.json -i --stream --max-tokens 1024 --prefix-cache
 *   ./demo/qwen3_infer model-sq.bin   tokenizer.json -i --stream --max-tokens 1024 --prefix-cache
 */

#include "model/qwen3.h"
#include "model/qwen3_awq.h"
#include "model/qwen3_sq.h"
#include "inference_common.h"

int main(int argc, char* argv[]) {
    inference::ModelInferConfig model_config;
    model_config.skip_tokens = {151645};  // EOS only
    model_config.remove_thinking = true;  // Qwen3 支持 <think> 思考模式
    model_config.model_name = "Qwen3";

    if (argc >= 2 && model::is_awq_model_file(argv[1])) {
        model_config.model_name = "Qwen3-AWQ";
        return inference::run_model_inference<model::Qwen3AWQModel>(
            argc, argv,
            "Qwen3 AWQ INT4 Model Inference with Multi-Turn Dialog and RadixTree PrefixCache",
            model_config,
            true  // Qwen3 默认启用 CUDA Graph
        );
    }

    if (argc >= 2 && model::is_sq_model_file(argv[1])) {
        model_config.model_name = "Qwen3-SQ";
        return inference::run_model_inference<model::Qwen3SQModel>(
            argc, argv,
            "Qwen3 SmoothQuant INT8 Model Inference with Multi-Turn Dialog and RadixTree PrefixCache",
            model_config,
            true  // Qwen3 默认启用 CUDA Graph
        );
    }

    return inference::run_model_inference<model::Qwen3Model>(
        argc, argv,
        "Qwen3 Model Inference with Multi-Turn Dialog and RadixTree PrefixCache",
        model_config,
        true  // Qwen3 默认启用 CUDA Graph
    );
}
