/**
 * SmoothQuant INT8 GEMM — Optimized Dual-Path Architecture v2
 *
 * Improvements over v1:
 *   - Fixed race condition: replaced broken fused absmax+quantize with correct
 *     2-kernel approach (absmax reduction → quantize+alpha, separate launches)
 *   - __dp4a hardware intrinsic for INT8 dot products (4 MACs / 1 instruction)
 *   - 128-bit (int4) vectorized loads for maximum memory bandwidth
 *   - Shared input quantization API: quantize once, reuse for Q/K/V GEMV calls
 *     (saves 6 kernel launches per layer, 216 per decode step)
 *
 * PATH 1 — Decode (M=1): INT8 GEMV kernel (bandwidth-optimized)
 *   - absmax reduction + quantize (2 separate kernels, correct synchronization)
 *   - __dp4a INT8 dot product + int4 128-bit vectorized loads
 *   - Warp shuffle reduction, 8 outputs/block (256 threads)
 *
 * PATH 2 — Prefill (M>1): CUTLASS INT8 Tensor Core GEMM
 *   - Separate absmax + quantize (2 kernels)
 *   - 16×8×32 MMA instruction (SM80/SM87)
 *   - Adaptive tile: 128×128×64 for small M, 256×128×64 for large M
 *
 * All paths fully CUDA-graph-compatible (device-side alpha, monotonic buffers).
 */

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include "sq_gemm_kernel.cuh"

// CUTLASS headers (header-only library)
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/gemm/threadblock/threadblock_swizzle.h>

namespace kernel {

// ========================== CUTLASS INT8 GEMM Types =========================

// Large tile for big M (prefill with long sequences)
using CutlassInt8Gemm = cutlass::gemm::device::Gemm<
    int8_t, cutlass::layout::RowMajor,
    int8_t, cutlass::layout::ColumnMajor,
    cutlass::half_t, cutlass::layout::RowMajor,
    int32_t,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<256, 128, 64>,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<16, 8, 32>,
    cutlass::epilogue::thread::LinearCombination<
        cutlass::half_t, 8, int32_t, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    3>;

// Smaller tile for moderate M (small prefill batches)
using CutlassInt8GemmSmall = cutlass::gemm::device::Gemm<
    int8_t, cutlass::layout::RowMajor,
    int8_t, cutlass::layout::ColumnMajor,
    cutlass::half_t, cutlass::layout::RowMajor,
    int32_t,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 64>,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<16, 8, 32>,
    cutlass::epilogue::thread::LinearCombination<
        cutlass::half_t, 8, int32_t, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    3>;

// ========================= AbsMax Reduction Kernel ==========================
//
// Per-tensor absmax: computes max(|x_i|) across all elements.
// Each block does shared-memory tree reduction, then atomicMax to global.
// Correct with any number of blocks (no inter-block sync needed).
//
__global__ void sq_absmax_kernel(
    const half* __restrict__ input,
    int* __restrict__ d_max_as_int,
    int total_elements)
{
    extern __shared__ float sdata[];
    const int tid = threadIdx.x;
    const int gid = (blockIdx.x * blockDim.x + tid) * 4;

    float local_max = 0.0f;
    if (gid + 3 < total_elements) {
        const half2* h2 = reinterpret_cast<const half2*>(input + gid);
        half2 v0 = __ldg(h2);
        half2 v1 = __ldg(h2 + 1);
        float2 f0 = __half22float2(v0);
        float2 f1 = __half22float2(v1);
        local_max = fmaxf(fmaxf(fabsf(f0.x), fabsf(f0.y)),
                          fmaxf(fabsf(f1.x), fabsf(f1.y)));
    } else {
        for (int i = gid; i < total_elements && i < gid + 4; ++i) {
            local_max = fmaxf(local_max, fabsf(__half2float(input[i])));
        }
    }

    sdata[tid] = local_max;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicMax(d_max_as_int, __float_as_int(sdata[0]));
    }
}

// =================== Quantize + Compute Alpha Kernel ========================
//
// Reads finalized absmax from device memory (set by sq_absmax_kernel).
// Quantizes FP16→INT8 and computes alpha = (absmax/127) * weight_scale.
// Runs as a SEPARATE kernel after absmax to avoid inter-block race conditions.
//
__global__ void sq_quantize_and_alpha_kernel(
    const half* __restrict__ input_fp16,
    int8_t* __restrict__ output_int8,
    const int* __restrict__ d_max_as_int,
    float weight_scale,
    float* __restrict__ d_alpha,
    int total_elements)
{
    const float absmax = __int_as_float(*d_max_as_int);
    const float inv_scale = (absmax > 1e-6f) ? 127.0f / absmax : 0.0f;

    if (blockIdx.x == 0 && threadIdx.x == 0) {
        const float input_scale = (absmax > 1e-6f) ? absmax / 127.0f : 0.0f;
        *d_alpha = input_scale * weight_scale;
    }

    const int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;

    if (idx + 3 < total_elements) {
        const half2* h2 = reinterpret_cast<const half2*>(input_fp16 + idx);
        half2 v0 = __ldg(h2);
        half2 v1 = __ldg(h2 + 1);
        float2 f0 = __half22float2(v0);
        float2 f1 = __half22float2(v1);

        int i0 = max(-128, min(127, __float2int_rn(f0.x * inv_scale)));
        int i1 = max(-128, min(127, __float2int_rn(f0.y * inv_scale)));
        int i2 = max(-128, min(127, __float2int_rn(f1.x * inv_scale)));
        int i3 = max(-128, min(127, __float2int_rn(f1.y * inv_scale)));

        int32_t packed = (i0 & 0xFF) | ((i1 & 0xFF) << 8) |
                         ((i2 & 0xFF) << 16) | ((i3 & 0xFF) << 24);
        *reinterpret_cast<int32_t*>(output_int8 + idx) = packed;
    } else {
        for (int i = idx; i < total_elements && i < idx + 4; ++i) {
            float val = __half2float(input_fp16[i]) * inv_scale;
            output_int8[i] = static_cast<int8_t>(max(-128, min(127, __float2int_rn(val))));
        }
    }
}

// ========================= INT8 GEMV with __dp4a ============================
//
// Bandwidth-optimized GEMV for decode phase (M=1):
//   output[n] = alpha * sum_k(input_int8[k] * weight_int8[n, k])
//
// Key optimizations vs v1:
// - __dp4a: hardware INT8×4 dot product (4 MACs in 1 instruction, vs 12 in v1)
// - int4 (128-bit) vectorized loads: 16 INT8 per load (vs 4 per load in v1)
// - 256 threads/block = 8 warps, each warp handles 1 output channel
// - FP32 accumulator with warp shuffle reduction
//
__global__ __launch_bounds__(256, 4)
void sq_gemv_int8_kernel(
    const int8_t* __restrict__ input_int8,       // [K] quantized activation
    const int8_t* __restrict__ weight_int8,      // [N, K] row-major (each row = one output channel)
    half* __restrict__ output_fp16,              // [N]
    const float* __restrict__ d_alpha,           // device-side scale = input_scale * weight_scale
    int K,
    int N)
{
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;

    // Each warp processes one output channel
    const int out_idx = blockIdx.x * 8 + warp_id;
    if (out_idx >= N) return;

    const float alpha = *d_alpha;

    // Pointer to this output channel's weight row
    const int8_t* w_row = weight_int8 + static_cast<int64_t>(out_idx) * K;

    int32_t acc = 0;

    // Main loop: 128-bit loads + __dp4a (16 INT8 elements per iteration)
    const int num_vec16 = K / 16;
    const int4* input_i16 = reinterpret_cast<const int4*>(input_int8);
    const int4* weight_i16 = reinterpret_cast<const int4*>(w_row);

    #pragma unroll 4
    for (int i = lane_id; i < num_vec16; i += 32) {
        int4 x = __ldg(input_i16 + i);
        int4 w = __ldg(weight_i16 + i);
        acc = __dp4a(x.x, w.x, acc);
        acc = __dp4a(x.y, w.y, acc);
        acc = __dp4a(x.z, w.z, acc);
        acc = __dp4a(x.w, w.w, acc);
    }

    // Handle remainder (K not divisible by 16)
    const int base = num_vec16 * 16;
    for (int i = base + lane_id; i < K; i += 32) {
        acc += static_cast<int32_t>(input_int8[i]) * static_cast<int32_t>(w_row[i]);
    }

    // Warp reduction using shuffle
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        acc += __shfl_down_sync(0xffffffff, acc, offset);
    }

    // Lane 0 writes the dequantized FP16 output
    if (lane_id == 0) {
        output_fp16[out_idx] = __float2half(alpha * static_cast<float>(acc));
    }
}

// ======================== Pre-Quantized GEMV ================================
//
// GEMV with pre-quantized INT8 input. Reads input_scale from device memory
// and multiplies by per-layer weight_scale to form alpha.
//
// Used for shared quantization (QKV): quantize input once, then call this
// kernel 3 times with different weight matrices and weight_scales.
//
__global__ __launch_bounds__(256, 4)
void sq_gemv_preq_kernel(
    const int8_t* __restrict__ input_int8,       // [K] pre-quantized activation
    const int8_t* __restrict__ weight_int8,      // [N, K] row-major
    half* __restrict__ output_fp16,              // [N]
    const float* __restrict__ d_input_scale,     // device ptr: input_scale = absmax/127
    float weight_scale,                          // host constant: per-layer weight scale
    int K,
    int N)
{
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;

    const int out_idx = blockIdx.x * 8 + warp_id;
    if (out_idx >= N) return;

    const float alpha = (*d_input_scale) * weight_scale;

    const int8_t* w_row = weight_int8 + static_cast<int64_t>(out_idx) * K;

    int32_t acc = 0;

    const int num_vec16 = K / 16;
    const int4* input_i16 = reinterpret_cast<const int4*>(input_int8);
    const int4* weight_i16 = reinterpret_cast<const int4*>(w_row);

    #pragma unroll 4
    for (int i = lane_id; i < num_vec16; i += 32) {
        int4 x = __ldg(input_i16 + i);
        int4 w = __ldg(weight_i16 + i);
        acc = __dp4a(x.x, w.x, acc);
        acc = __dp4a(x.y, w.y, acc);
        acc = __dp4a(x.z, w.z, acc);
        acc = __dp4a(x.w, w.w, acc);
    }

    const int base = num_vec16 * 16;
    for (int i = base + lane_id; i < K; i += 32) {
        acc += static_cast<int32_t>(input_int8[i]) * static_cast<int32_t>(w_row[i]);
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        acc += __shfl_down_sync(0xffffffff, acc, offset);
    }

    if (lane_id == 0) {
        output_fp16[out_idx] = __float2half(alpha * static_cast<float>(acc));
    }
}

// ================== Fused SQ FFN GEMV (Gate + Up + SwiGLU) ==================
//
// Fuses 3 operations for decode (M=1):
//   gate = W1[row,:] · input_int8     (INT8 dot product)
//   up   = W3[row,:] · input_int8     (INT8 dot product)
//   output[row] = SiLU(alpha_w1 * gate) * (alpha_w3 * up)
//
// Where alpha_wX = input_scale * wX_weight_scale
//       input_scale is read from device memory (d_input_scale)
//       wX_weight_scale is passed as host-side float params
//
// Saves 2 SQ GEMM calls (= 6 kernel launches) + eliminates intermediate buffers.
//
// Uses __dp4a + int4 vectorized loads for maximum throughput.
// Each warp handles one output row. 8 warps/block = 8 rows/block.
//
__global__ __launch_bounds__(256, 4)
void sq_fused_ffn_gemv_kernel(
    const int8_t* __restrict__ input_int8,       // [K] quantized activation
    const int8_t* __restrict__ w1_int8,          // [hidden_dim, K] gate weight
    const int8_t* __restrict__ w3_int8,          // [hidden_dim, K] up weight
    half* __restrict__ output_fp16,              // [hidden_dim]
    const float* __restrict__ d_input_scale,     // device-side input_scale (= absmax/127)
    float w1_weight_scale,                       // host-side constant
    float w3_weight_scale,                       // host-side constant
    int K,
    int hidden_dim)
{
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;

    const int row = blockIdx.x * 8 + warp_id;
    if (row >= hidden_dim) return;

    const float input_scale = *d_input_scale;
    const float alpha_w1 = input_scale * w1_weight_scale;
    const float alpha_w3 = input_scale * w3_weight_scale;

    const int8_t* w1_row = w1_int8 + static_cast<int64_t>(row) * K;
    const int8_t* w3_row = w3_int8 + static_cast<int64_t>(row) * K;

    int32_t acc_gate = 0;
    int32_t acc_up = 0;

    const int num_vec16 = K / 16;
    const int4* input_i16 = reinterpret_cast<const int4*>(input_int8);
    const int4* w1_i16 = reinterpret_cast<const int4*>(w1_row);
    const int4* w3_i16 = reinterpret_cast<const int4*>(w3_row);

    #pragma unroll 4
    for (int i = lane_id; i < num_vec16; i += 32) {
        int4 x = __ldg(input_i16 + i);
        int4 g = __ldg(w1_i16 + i);
        int4 u = __ldg(w3_i16 + i);

        acc_gate = __dp4a(x.x, g.x, acc_gate);
        acc_gate = __dp4a(x.y, g.y, acc_gate);
        acc_gate = __dp4a(x.z, g.z, acc_gate);
        acc_gate = __dp4a(x.w, g.w, acc_gate);

        acc_up = __dp4a(x.x, u.x, acc_up);
        acc_up = __dp4a(x.y, u.y, acc_up);
        acc_up = __dp4a(x.z, u.z, acc_up);
        acc_up = __dp4a(x.w, u.w, acc_up);
    }

    // Remainder (K not divisible by 16)
    const int base = num_vec16 * 16;
    for (int i = base + lane_id; i < K; i += 32) {
        int8_t x = input_int8[i];
        acc_gate += static_cast<int32_t>(x) * static_cast<int32_t>(w1_row[i]);
        acc_up   += static_cast<int32_t>(x) * static_cast<int32_t>(w3_row[i]);
    }

    // Warp reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        acc_gate += __shfl_down_sync(0xffffffff, acc_gate, offset);
        acc_up   += __shfl_down_sync(0xffffffff, acc_up, offset);
    }

    if (lane_id == 0) {
        float gate = alpha_w1 * static_cast<float>(acc_gate);
        float up   = alpha_w3 * static_cast<float>(acc_up);
        // SiLU(gate) * up
        float gate_activated = gate / (1.0f + __expf(-gate));
        output_fp16[row] = __float2half(gate_activated * up);
    }
}

// ======================== Workspace Management ==============================
struct SQWorkspace {
    int8_t* input_int8 = nullptr;
    int*    max_int    = nullptr;
    float*  alpha      = nullptr;   // Stores alpha or input_scale depending on context
    size_t  input_cap  = 0;

    void ensure(size_t need) {
        if (need > input_cap) {
            if (input_int8) cudaFree(input_int8);
            input_cap = need * 2;
            cudaMalloc(&input_int8, input_cap);
        }
        if (!max_int) {
            cudaMalloc(&max_int, sizeof(int));
            cudaMalloc(&alpha, sizeof(float));
        }
    }
};

static SQWorkspace g_workspace;

// ========================= M=1 GEMV Dispatch ===============================
//
// Correct 3-kernel pipeline (no inter-block race conditions):
//   1. cudaMemsetAsync (reset absmax)
//   2. sq_absmax_kernel (parallel block reduction → atomicMax)
//   3. sq_quantize_and_alpha_kernel (reads finalized absmax, quantizes + alpha)
//   4. sq_gemv_int8_kernel (__dp4a GEMV with 128-bit loads)
//
static void sq_gemv_m1(
    const half* input_fp16,
    const int8_t* qweight,
    half* output_fp16,
    float weight_scale,
    int K,
    int N,
    cudaStream_t stream)
{
    g_workspace.ensure(static_cast<size_t>(K));

    constexpr int kThreads = 256;
    int quant_blocks = (K + kThreads * 4 - 1) / (kThreads * 4);

    // Reset absmax accumulator
    cudaMemsetAsync(g_workspace.max_int, 0, sizeof(int), stream);

    // Phase 1: AbsMax reduction
    sq_absmax_kernel<<<quant_blocks, kThreads, kThreads * sizeof(float), stream>>>(
        input_fp16, g_workspace.max_int, K);

    // Phase 2: Quantize FP16→INT8 + compute alpha = input_scale * weight_scale
    sq_quantize_and_alpha_kernel<<<quant_blocks, kThreads, 0, stream>>>(
        input_fp16, g_workspace.input_int8, g_workspace.max_int,
        weight_scale, g_workspace.alpha, K);

    // Phase 3: INT8 GEMV with __dp4a
    int gemv_blocks = (N + 7) / 8;
    sq_gemv_int8_kernel<<<gemv_blocks, 256, 0, stream>>>(
        g_workspace.input_int8, qweight, output_fp16,
        g_workspace.alpha, K, N);
}

// ========================= M>1 CUTLASS GEMM Path ============================
static void sq_gemm_cutlass(
    const half* input_fp16,
    const int8_t* qweight,
    half* output_fp16,
    float weight_scale,
    int M,
    int K,
    int N,
    cudaStream_t stream)
{
    const int input_elements = M * K;
    g_workspace.ensure(static_cast<size_t>(input_elements));

    // Reset absmax accumulator
    cudaMemsetAsync(g_workspace.max_int, 0, sizeof(int), stream);

    constexpr int kThreads = 256;
    int blocks = (input_elements + kThreads * 4 - 1) / (kThreads * 4);

    // Phase 1: Compute per-tensor absmax
    sq_absmax_kernel<<<blocks, kThreads, kThreads * sizeof(float), stream>>>(
        input_fp16, g_workspace.max_int, input_elements);

    // Phase 2: Quantize FP16→INT8 + compute alpha
    sq_quantize_and_alpha_kernel<<<blocks, kThreads, 0, stream>>>(
        input_fp16, g_workspace.input_int8, g_workspace.max_int,
        weight_scale, g_workspace.alpha, input_elements);

    // Phase 3: CUTLASS INT8 Tensor Core GEMM
    auto input_size  = cutlass::MatrixCoord(M, K);
    auto weight_size = cutlass::MatrixCoord(K, N);
    auto output_size = cutlass::MatrixCoord(M, N);

    cutlass::TensorRef<int8_t, cutlass::layout::RowMajor> input_ref(
        g_workspace.input_int8,
        cutlass::layout::RowMajor::packed(input_size));

    cutlass::TensorRef<int8_t, cutlass::layout::ColumnMajor> weight_ref(
        const_cast<int8_t*>(qweight),
        cutlass::layout::ColumnMajor::packed(weight_size));

    cutlass::TensorRef<cutlass::half_t, cutlass::layout::RowMajor> output_ref(
        reinterpret_cast<cutlass::half_t*>(output_fp16),
        cutlass::layout::RowMajor::packed(output_size));

    // Use smaller tile for small M to improve efficiency
    if (M <= 32) {
        typename CutlassInt8GemmSmall::EpilogueOutputOp::Params epilogue_params(
            g_workspace.alpha, nullptr);

        cutlass::gemm::GemmCoord problem_size(M, N, K);
        typename CutlassInt8GemmSmall::Arguments arguments{
            problem_size, input_ref, weight_ref, output_ref, output_ref,
            epilogue_params, 1};

        CutlassInt8GemmSmall gemm_op;
        cutlass::Status status = gemm_op.can_implement(arguments);
        if (status == cutlass::Status::kSuccess) {
            status = gemm_op.initialize(arguments, nullptr, stream);
            if (status == cutlass::Status::kSuccess) {
                gemm_op(stream);
                return;
            }
        }
    }

    // Large tile for big M
    typename CutlassInt8Gemm::EpilogueOutputOp::Params epilogue_params(
        g_workspace.alpha, nullptr);

    cutlass::gemm::GemmCoord problem_size(M, N, K);
    typename CutlassInt8Gemm::Arguments arguments{
        problem_size, input_ref, weight_ref, output_ref, output_ref,
        epilogue_params, 1};

    CutlassInt8Gemm gemm_op;
    cutlass::Status status = gemm_op.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) return;

    status = gemm_op.initialize(arguments, nullptr, stream);
    if (status != cutlass::Status::kSuccess) return;

    gemm_op(stream);
}

// ========================= Main SQ GEMM Entry ==============================
void sq_gemm_cu(
    const half* input_fp16,
    const int8_t* qweight,
    half* output_fp16,
    float weight_scale,
    int batch_size,
    int in_features,
    int out_features,
    cudaStream_t stream)
{
    const int M = batch_size;
    const int K = in_features;
    const int N = out_features;

    if (M == 1) {
        // Decode path: bandwidth-optimized INT8 GEMV
        sq_gemv_m1(input_fp16, qweight, output_fp16, weight_scale, K, N, stream);
    } else {
        // Prefill path: CUTLASS INT8 Tensor Core GEMM
        sq_gemm_cutlass(input_fp16, qweight, output_fp16, weight_scale, M, K, N, stream);
    }
}

// ======================= Fused SQ FFN Entry =================================
//
// For decode (M=1): quantize input once, then do fused W1+W3 GEMV + SwiGLU.
// Saves 6 kernel launches compared to separate w1 + w3 SQ GEMM calls.
//
// Pipeline:
//   1. cudaMemsetAsync (reset absmax)
//   2. sq_absmax_kernel (absmax reduction)
//   3. sq_quantize_and_alpha_kernel (quantize + input_scale, weight_scale=1.0)
//   4. sq_fused_ffn_gemv_kernel (W1+W3 GEMV + SwiGLU with per-layer weight_scales)
//
void sq_fused_ffn_cu(
    const half* input_fp16,
    const int8_t* w1_int8,
    const int8_t* w3_int8,
    half* output_fp16,
    float w1_weight_scale,
    float w3_weight_scale,
    int in_features,
    int hidden_dim,
    cudaStream_t stream)
{
    const int K = in_features;

    // Ensure workspace for input quantization
    g_workspace.ensure(static_cast<size_t>(K));

    constexpr int kThreads = 256;
    int quant_blocks = (K + kThreads * 4 - 1) / (kThreads * 4);

    // Reset absmax accumulator
    cudaMemsetAsync(g_workspace.max_int, 0, sizeof(int), stream);

    // Phase 1: AbsMax reduction
    sq_absmax_kernel<<<quant_blocks, kThreads, kThreads * sizeof(float), stream>>>(
        input_fp16, g_workspace.max_int, K);

    // Phase 2: Quantize + compute input_scale
    // Using weight_scale=1.0 so alpha = input_scale * 1.0 = input_scale
    sq_quantize_and_alpha_kernel<<<quant_blocks, kThreads, 0, stream>>>(
        input_fp16, g_workspace.input_int8, g_workspace.max_int,
        1.0f, g_workspace.alpha, K);

    // Phase 3: Fused W1+W3 GEMV + SwiGLU
    int ffn_blocks = (hidden_dim + 7) / 8;
    sq_fused_ffn_gemv_kernel<<<ffn_blocks, 256, 0, stream>>>(
        g_workspace.input_int8, w1_int8, w3_int8, output_fp16,
        g_workspace.alpha,      // d_input_scale = absmax/127
        w1_weight_scale,        // host-side W1 scale
        w3_weight_scale,        // host-side W3 scale
        K, hidden_dim);
}

// =================== Shared Quantization API ================================
//
// Quantize input once, then reuse for multiple GEMV calls with different weights.
// Designed for QKV projections where all 3 use the same input (rms_out).
//
// Saves 2 × (memset + absmax + quantize) = 6 kernel launches per layer.
// For 36 layers: 216 fewer kernel launches per decode step.
//
// After sq_quantize_input_cu():
//   g_workspace.input_int8 = quantized input [K]
//   g_workspace.alpha = input_scale = absmax / 127
//
void sq_quantize_input_cu(
    const half* input_fp16,
    int K,
    cudaStream_t stream)
{
    g_workspace.ensure(static_cast<size_t>(K));

    constexpr int kThreads = 256;
    int blocks = (K + kThreads * 4 - 1) / (kThreads * 4);

    // Reset absmax accumulator
    cudaMemsetAsync(g_workspace.max_int, 0, sizeof(int), stream);

    // AbsMax reduction
    sq_absmax_kernel<<<blocks, kThreads, kThreads * sizeof(float), stream>>>(
        input_fp16, g_workspace.max_int, K);

    // Quantize + compute input_scale (weight_scale=1.0 → alpha = input_scale)
    sq_quantize_and_alpha_kernel<<<blocks, kThreads, 0, stream>>>(
        input_fp16, g_workspace.input_int8, g_workspace.max_int,
        1.0f, g_workspace.alpha, K);
}

// GEMV with pre-quantized input from workspace.
// Reads input_scale from g_workspace.alpha and multiplies by per-layer weight_scale.
void sq_gemv_preq_cu(
    const int8_t* qweight,
    half* output_fp16,
    float weight_scale,
    int K,
    int N,
    cudaStream_t stream)
{
    int blocks = (N + 7) / 8;
    sq_gemv_preq_kernel<<<blocks, 256, 0, stream>>>(
        g_workspace.input_int8, qweight, output_fp16,
        g_workspace.alpha, weight_scale, K, N);
}

}  // namespace kernel
