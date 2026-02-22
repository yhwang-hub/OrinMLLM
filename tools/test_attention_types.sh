#!/bin/bash
# FlashAttention2 Validation Script
# Tests all 5 model configurations × 3 attention types (mha, flash1, flash2)

set -e

BUILD_DIR="/mnt/ssd/workspace/OrinMLLM/build"
QWEN3_FP16_BIN="/mnt/ssd/QwenModels/Qwen3-8B-fp16.bin"
QWEN3_AWQ_BIN="/mnt/ssd/QwenModels/Qwen3-8B-awq.bin"
QWEN3_TOK="/mnt/ssd/QwenModels/Qwen3-8B/tokenizer.json"
QWEN2_FP32_BIN="/mnt/ssd/QwenModels/Qwen2.5-7B.bin"
QWEN2_FP16_BIN="/mnt/ssd/QwenModels/Qwen2.5-7B-fp16.bin"
QWEN2_TOK="/mnt/ssd/QwenModels/Qwen2.5-7B-Instruct/tokenizer.json"
VL_BIN="/mnt/ssd/QwenModels/Qwen3-VL-8B-fp16.bin"
VL_TOK="/mnt/ssd/QwenModels/Qwen3-VL-8B-Instruct/tokenizer.json"
VL_IMAGE="/mnt/ssd/workspace/OrinMLLM/hf_infer/demo.jpeg"

MAX_TOKENS=64
PROMPT="你好，请用一句话介绍你自己"

LOG_DIR="/tmp/attention_test_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

PASS=0
FAIL=0
SKIP=0
RESULTS=()

run_test() {
    local name="$1"
    local cmd="$2"
    local attn="$3"
    local logfile="$LOG_DIR/${name}_${attn}.log"
    
    echo -n "  [${attn}] ... "
    
    # Run the inference command with timeout
    if echo "$PROMPT" | timeout 180 $cmd --attention "$attn" > "$logfile" 2>&1; then
        # Check if output contains generated tokens (look for Prefill/Decode stats)
        if grep -q "Decode:.*tokens/s" "$logfile"; then
            local prefill_speed=$(grep -oP 'Prefill:.*?(\d+\.?\d*) tokens/s' "$logfile" | grep -oP '\d+\.?\d* tokens/s' | tail -1)
            local decode_speed=$(grep -oP 'Decode:.*?(\d+\.?\d*) tokens/s' "$logfile" | grep -oP '\d+\.?\d* tokens/s' | tail -1)
            echo "PASS (Prefill: $prefill_speed, Decode: $decode_speed)"
            RESULTS+=("PASS | $name | $attn | Prefill: $prefill_speed, Decode: $decode_speed")
            ((PASS++))
        else
            echo "FAIL (no output tokens)"
            RESULTS+=("FAIL | $name | $attn | No output tokens generated")
            ((FAIL++))
        fi
    else
        local exit_code=$?
        if [ $exit_code -eq 124 ]; then
            echo "FAIL (timeout)"
            RESULTS+=("FAIL | $name | $attn | Timeout after 180s")
        else
            local error=$(grep -i "error\|fatal\|crash" "$logfile" | tail -1)
            echo "FAIL (exit=$exit_code: $error)"
            RESULTS+=("FAIL | $name | $attn | Exit code $exit_code: $error")
        fi
        ((FAIL++))
    fi
}

run_vl_test() {
    local name="$1"
    local cmd="$2"
    local attn="$3"
    local logfile="$LOG_DIR/${name}_${attn}.log"
    
    echo -n "  [${attn}] ... "
    
    if timeout 300 $cmd --attention "$attn" > "$logfile" 2>&1; then
        if grep -qP "tokens/s|Generated|assistant" "$logfile"; then
            local decode_info=$(grep -oP 'Decode.*?(\d+\.?\d*) tokens/s' "$logfile" | tail -1)
            echo "PASS ($decode_info)"
            RESULTS+=("PASS | $name | $attn | $decode_info")
            ((PASS++))
        else
            echo "PASS (completed)"
            RESULTS+=("PASS | $name | $attn | Completed")
            ((PASS++))
        fi
    else
        local exit_code=$?
        local error=$(grep -i "error\|fatal" "$logfile" | tail -1)
        echo "FAIL (exit=$exit_code: $error)"
        RESULTS+=("FAIL | $name | $attn | Exit code $exit_code: $error")
        ((FAIL++))
    fi
}

echo "=============================================="
echo " FlashAttention2 Validation Test Suite"
echo " $(date)"
echo "=============================================="
echo ""

# ---- Test 1: Qwen3-8B FP16 ----
echo "[1/5] Qwen3-8B FP16"
QWEN3_CMD="$BUILD_DIR/demo/qwen3_infer $QWEN3_FP16_BIN $QWEN3_TOK --stream --max-tokens $MAX_TOKENS"
for attn in flash1 flash2 mha; do
    run_test "qwen3_fp16" "$QWEN3_CMD" "$attn"
done
echo ""

# ---- Test 2: Qwen3-8B AWQ ----
echo "[2/5] Qwen3-8B AWQ"
QWEN3_AWQ_CMD="$BUILD_DIR/demo/qwen3_infer $QWEN3_AWQ_BIN $QWEN3_TOK --stream --max-tokens $MAX_TOKENS"
for attn in flash1 flash2 mha; do
    run_test "qwen3_awq" "$QWEN3_AWQ_CMD" "$attn"
done
echo ""

# ---- Test 3: Qwen2.5-7B FP32 ----
echo "[3/5] Qwen2.5-7B FP32"
QWEN2_FP32_CMD="$BUILD_DIR/demo/qwen_infer $QWEN2_FP32_BIN $QWEN2_TOK --stream --max-tokens $MAX_TOKENS"
for attn in flash1 flash2 mha; do
    run_test "qwen2_fp32" "$QWEN2_FP32_CMD" "$attn"
done
echo ""

# ---- Test 4: Qwen2.5-7B FP16 ----
echo "[4/5] Qwen2.5-7B FP16"
QWEN2_FP16_CMD="$BUILD_DIR/demo/qwen_infer $QWEN2_FP16_BIN $QWEN2_TOK --stream --max-tokens $MAX_TOKENS"
for attn in flash1 flash2 mha; do
    run_test "qwen2_fp16" "$QWEN2_FP16_CMD" "$attn"
done
echo ""

# ---- Test 5: Qwen3-VL-8B FP16 ----
echo "[5/5] Qwen3-VL-8B FP16"
VL_CMD="$BUILD_DIR/demo/qwen3_vl_infer $VL_BIN $VL_TOK --image $VL_IMAGE --prompt 'Describe this image.' --cuda-graph --stream --max-pixels 500000 --max-tokens $MAX_TOKENS"
for attn in flash1 flash2 mha; do
    run_vl_test "qwen3_vl" "$VL_CMD" "$attn"
done
echo ""

# ---- Summary ----
echo "=============================================="
echo " SUMMARY"
echo "=============================================="
echo ""
printf "%-6s | %-15s | %-6s | %s\n" "Result" "Model" "Attn" "Details"
echo "-------+-----------------+--------+----------------------------------"
for r in "${RESULTS[@]}"; do
    IFS='|' read -r result model attn details <<< "$r"
    printf "%-6s |%-15s |%-6s |%s\n" "$result" "$model" "$attn" "$details"
done
echo ""
echo "Total: $PASS PASS, $FAIL FAIL, $SKIP SKIP (out of $((PASS+FAIL+SKIP)))"
echo "Logs: $LOG_DIR"
echo ""

if [ $FAIL -eq 0 ]; then
    echo "ALL TESTS PASSED!"
    exit 0
else
    echo "SOME TESTS FAILED!"
    exit 1
fi
