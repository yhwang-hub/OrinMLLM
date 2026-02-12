#!/bin/bash
# Collect performance data for both reference and optimized implementations

REF_BIN="/mnt/ssd/workspace/KuiperLLama_20260202_fp16_awq_vlm_Refactor/build/demo"
OPT_BIN="/mnt/ssd/workspace/OrinMLLM/build/demo"
OUTDIR="/tmp/perf_comparison"
mkdir -p $OUTDIR

echo "=== Performance Comparison: Reference vs Optimized ==="
echo ""

# Function to run inference and extract perf
run_and_extract() {
    local label="$1"
    local cmd="$2"
    local outfile="$3"
    echo "Running: $label"
    eval "$cmd" > "$outfile" 2>&1
    grep -i "prefill\|decode" "$outfile" | tail -2
    echo ""
}

# Qwen3-8B-fp16
echo "--- Qwen3-8B-fp16 ---"
run_and_extract "Reference" "echo '你好' | $REF_BIN/qwen3_infer /mnt/ssd/QwenModels/Qwen3-8B-fp16.bin /mnt/ssd/QwenModels/Qwen3-8B/tokenizer.json --stream --max-tokens 128 --prefix-cache -i" "$OUTDIR/ref_q3fp16.txt"
run_and_extract "Optimized" "echo '你好' | $OPT_BIN/qwen3_infer /mnt/ssd/QwenModels/Qwen3-8B-fp16.bin /mnt/ssd/QwenModels/Qwen3-8B/tokenizer.json --stream --max-tokens 128 --prefix-cache -i" "$OUTDIR/opt_q3fp16.txt"

# Qwen3-8B-AWQ
echo "--- Qwen3-8B-AWQ ---"
run_and_extract "Reference" "echo '你好' | $REF_BIN/qwen3_infer /mnt/ssd/QwenModels/Qwen3-8B-awq.bin /mnt/ssd/QwenModels/Qwen3-8B/tokenizer.json --stream --max-tokens 128 --prefix-cache -i" "$OUTDIR/ref_q3awq.txt"
run_and_extract "Optimized" "echo '你好' | $OPT_BIN/qwen3_infer /mnt/ssd/QwenModels/Qwen3-8B-awq.bin /mnt/ssd/QwenModels/Qwen3-8B/tokenizer.json --stream --max-tokens 128 --prefix-cache -i" "$OUTDIR/opt_q3awq.txt"

# Qwen2.5-7B (INT8)
echo "--- Qwen2.5-7B ---"
run_and_extract "Reference" "echo '你好' | $REF_BIN/qwen_infer /mnt/ssd/QwenModels/Qwen2.5-7B.bin /mnt/ssd/QwenModels/Qwen2.5-7B-Instruct/tokenizer.json --stream --max-tokens 128 --prefix-cache -i" "$OUTDIR/ref_q25.txt"
run_and_extract "Optimized" "echo '你好' | $OPT_BIN/qwen_infer /mnt/ssd/QwenModels/Qwen2.5-7B.bin /mnt/ssd/QwenModels/Qwen2.5-7B-Instruct/tokenizer.json --stream --max-tokens 128 --prefix-cache -i" "$OUTDIR/opt_q25.txt"

# Qwen2.5-7B-fp16
echo "--- Qwen2.5-7B-fp16 ---"
run_and_extract "Reference" "echo '你好' | $REF_BIN/qwen_infer /mnt/ssd/QwenModels/Qwen2.5-7B-fp16.bin /mnt/ssd/QwenModels/Qwen2.5-7B-Instruct/tokenizer.json --stream --max-tokens 128 --prefix-cache -i" "$OUTDIR/ref_q25fp16.txt"
run_and_extract "Optimized" "echo '你好' | $OPT_BIN/qwen_infer /mnt/ssd/QwenModels/Qwen2.5-7B-fp16.bin /mnt/ssd/QwenModels/Qwen2.5-7B-Instruct/tokenizer.json --stream --max-tokens 128 --prefix-cache -i" "$OUTDIR/opt_q25fp16.txt"

echo "=== All Tests Complete ==="
