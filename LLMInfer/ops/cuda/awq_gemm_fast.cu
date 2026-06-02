/*
 * AWQ Fast W4A16 GEMV Kernel - Decode Optimized (M=1)
 *
 * This kernel is specifically optimized for the decode phase (M=1) of LLM
 * inference, where the operation is a GEMV (matrix-vector multiply).
 *
 * ==================== WHY GEMV, NOT TENSOR CORE MMA? ====================
 *
 * For M=1 decode, GEMV is strictly better than Tensor Core MMA because:
 * 1. M=1 is memory-bandwidth-bound (compute intensity ~4 FLOPs/byte,
 *    well below the Orin roofline of ~49 FLOPs/byte)
 * 2. MMA m16n8k16 instructions require padding M=1→16, wasting 93.75%
 *    of compute while providing zero bandwidth benefit
 * 3. GEMV with 256 threads/block achieves 8x higher occupancy than
 *    MMA-based kernels (32 warps/SM vs 4 warps/SM), critical for
 *    hiding memory latency on bandwidth-bound operations
 *
 * ==================== LOP3 DEQUANTIZATION ====================
 *
 * We adopt the vllm-style LOP3 bit manipulation technique for INT4→FP16
 * conversion, which is 2-3x more instruction-efficient than scalar
 * bit extraction (shift + mask + cast).
 *
 * AWQ bit layout in INT32 (packing order {0,4,1,5,2,6,3,7}):
 *   bits[0:3]   = elem 0    bits[16:19] = elem 1
 *   bits[4:7]   = elem 2    bits[20:23] = elem 3
 *   bits[8:11]  = elem 4    bits[24:27] = elem 5
 *   bits[12:15] = elem 6    bits[28:31] = elem 7
 *
 * This layout naturally groups (even, odd) element pairs into the lower
 * and upper 16-bit halves, making it directly compatible with LOP3 half2
 * extraction — no weight repacking needed.
 *
 * LOP3 extraction:
 *   BOTTOM_MASK (0x000f000f): bits[0:3,16:19]  → half2{elem0, elem1}
 *   TOP_MASK    (0x00f000f0): bits[4:7,20:23]  → half2{elem2, elem3}
 *   (shift>>8 + BOTTOM_MASK): bits[8:11,24:27] → half2{elem4, elem5}
 *   (shift>>8 + TOP_MASK):    bits[12:15,28:31]→ half2{elem6, elem7}
 */

#include "awq_gemm_fast.cuh"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace kernel {

// =============================================================================
// LOP3-based INT4 to FP16 Extraction
// =============================================================================

/**
 * Extract 8 INT4 values from a packed INT32 into 4 half2 pairs using LOP3.
 *
 * The LOP3 instruction computes d = (a & b) | c in a single cycle,
 * which simultaneously masks the INT4 nibble and ORs it with the FP16
 * magic number to form a valid FP16 encoding.
 *
 * Output pairs follow AWQ element order: (0,1), (2,3), (4,5), (6,7).
 * Each output value is in the range [0, 15] as FP16.
 */
__device__ __forceinline__ void lop3_extract_int4_to_fp16x2(
    uint32_t packed,
    uint32_t* out  // 4 x uint32_t, each interpreted as half2
) {
    constexpr uint32_t BOTTOM_MASK = 0x000f000f;
    constexpr uint32_t TOP_MASK    = 0x00f000f0;
    constexpr uint32_t FP16_MAGIC  = 0x64006400;  // half2{1024.0, 1024.0}
    constexpr uint32_t ONE_16TH    = 0x2c002c00;   // half2{1/16, 1/16}
    constexpr uint32_t NEG_64      = 0xd400d400;   // half2{-64, -64}

    const uint32_t packed_hi = packed >> 8;

    // LOP3: d = (packed & mask) | magic  →  FP16 encoding of INT4 nibbles
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
        : "=r"(out[0]) : "r"(packed),    "n"(BOTTOM_MASK), "n"(FP16_MAGIC), "n"(0xea));
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
        : "=r"(out[1]) : "r"(packed),    "n"(TOP_MASK),    "n"(FP16_MAGIC), "n"(0xea));
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
        : "=r"(out[2]) : "r"(packed_hi), "n"(BOTTOM_MASK), "n"(FP16_MAGIC), "n"(0xea));
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
        : "=r"(out[3]) : "r"(packed_hi), "n"(TOP_MASK),    "n"(FP16_MAGIC), "n"(0xea));

    // Convert to proper FP16 integer values [0..15]:
    //   BOTTOM pairs: value = encoded - 1024.0
    //   TOP pairs:    value = encoded * (1/16) + (-64)  [undo 4-bit left shift]
    asm volatile("sub.f16x2 %0, %1, %2;\n"
        : "=r"(out[0]) : "r"(out[0]), "r"(FP16_MAGIC));
    asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n"
        : "=r"(out[1]) : "r"(out[1]), "r"(ONE_16TH), "r"(NEG_64));
    asm volatile("sub.f16x2 %0, %1, %2;\n"
        : "=r"(out[2]) : "r"(out[2]), "r"(FP16_MAGIC));
    asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n"
        : "=r"(out[3]) : "r"(out[3]), "r"(ONE_16TH), "r"(NEG_64));
}

// =============================================================================
// Coalesced GEMV Kernel (uses transposed qweight [N/8, K])
// =============================================================================
/**
 * Same algorithm as awq_gemv_fast_kernel but with coalesced qweight access
 * and vectorized uint4 loads for maximum bandwidth utilization.
 *
 * Original: qweight[k_idx * packed_N + packed_out_idx]
 *   → 32 lanes stride by packed_N * 4 bytes (2048-6144 bytes) → 32 cache lines/warp
 *
 * Transposed + vectorized: qweight_t[packed_out_idx * K + k_idx], uint4 loads
 *   → 32 lanes × 16 bytes = 512 bytes per iteration → 4 cache lines, fully coalesced
 *   → 4 K-positions per thread per iteration (128 per warp = 1 group for g=128)
 */
__global__ __launch_bounds__(256, 4)
void awq_gemv_coalesced_kernel(
    const half* __restrict__ X,              // [K]
    const int32_t* __restrict__ qweight_t,   // [N/8, K] transposed
    const int32_t* __restrict__ qzeros,      // [K/G, N/8]
    const half* __restrict__ scales,         // [K/G, N]
    half* __restrict__ Y,                    // [N]
    int K,
    int N,
    int group_size
) {
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;

    const int packed_out_idx = blockIdx.x * 8 + warp_id;
    const int out_base = packed_out_idx * 8;

    if (out_base >= N) return;

    const int packed_N = N / 8;
    const int n_groups = K / group_size;

    // Base pointer for this warp's row in the transposed layout
    const int32_t* warp_qweight = qweight_t + packed_out_idx * K;

    float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};

    for (int g = 0; g < n_groups; g++) {
        const uint32_t qz = static_cast<uint32_t>(__ldg(&qzeros[g * packed_N + packed_out_idx]));
        uint32_t z_h[4];
        lop3_extract_int4_to_fp16x2(qz, z_h);

        const uint4 scale_vec = *reinterpret_cast<const uint4*>(&scales[g * N + out_base]);
        const half2* s_h2 = reinterpret_cast<const half2*>(&scale_vec);

        half2 neg_sz_h2[4];
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            half2 z_h2 = *reinterpret_cast<const half2*>(&z_h[j]);
            neg_sz_h2[j] = __hneg2(__hmul2(s_h2[j], z_h2));
        }

        const int group_start = g * group_size;

        // Vectorized: each lane loads 4 consecutive int32 via uint4
        // 32 lanes × 4 = 128 = group_size, so 1 iteration per group
        for (int k = lane_id * 4; k < group_size; k += 128) {
            const int k_idx = group_start + k;

            // Load 4 packed weights (16 bytes, coalesced)
            const uint4 w4 = *reinterpret_cast<const uint4*>(&warp_qweight[k_idx]);

            // Load 4 input values (8 bytes)
            const uint2 x2 = *reinterpret_cast<const uint2*>(&X[k_idx]);
            const half* x_ptr = reinterpret_cast<const half*>(&x2);

            // Process 4 K-positions with ILP
            const uint32_t w_arr[4] = {w4.x, w4.y, w4.z, w4.w};

            #pragma unroll
            for (int v = 0; v < 4; v++) {
                const half2 x_h2 = __half2half2(x_ptr[v]);
                uint32_t w_h[4];
                lop3_extract_int4_to_fp16x2(w_arr[v], w_h);

                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    half2 w_h2 = *reinterpret_cast<const half2*>(&w_h[j]);
                    half2 dq_h2 = __hfma2(s_h2[j], w_h2, neg_sz_h2[j]);
                    half2 prod = __hmul2(x_h2, dq_h2);
                    acc[j * 2]     += __low2float(prod);
                    acc[j * 2 + 1] += __high2float(prod);
                }
            }
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            acc[i] += __shfl_down_sync(0xffffffff, acc[i], offset);
        }
    }

    if (lane_id == 0) {
        half out_half[8];
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            out_half[i] = __float2half(acc[i]);
        }
        *reinterpret_cast<uint4*>(&Y[out_base]) = *reinterpret_cast<uint4*>(out_half);
    }
}

void awq_gemv_coalesced_cu(
    const half* input,
    const int32_t* qweight_t,
    const int32_t* qzeros,
    const half* scales,
    half* output,
    int K,
    int N,
    int group_size,
    cudaStream_t stream
) {
    const int num_blocks = (N + 63) / 64;
    awq_gemv_coalesced_kernel<<<num_blocks, 256, 0, stream>>>(
        input, qweight_t, qzeros, scales, output,
        K, N, group_size
    );
}

// =============================================================================
// Transpose Kernel: [K, packed_N] → [packed_N, K]
// =============================================================================
__global__ void transpose_qweight_kernel(
    const int32_t* __restrict__ src,  // [K, packed_N]
    int32_t* __restrict__ dst,        // [packed_N, K]
    int K,
    int packed_N
) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int n = blockIdx.y * blockDim.y + threadIdx.y;
    if (k < K && n < packed_N) {
        dst[n * K + k] = src[k * packed_N + n];
    }
}

void awq_transpose_qweight_cu(
    const int32_t* src,
    int32_t* dst,
    int K,
    int packed_N,
    cudaStream_t stream
) {
    dim3 block(32, 8);
    dim3 grid((K + block.x - 1) / block.x, (packed_N + block.y - 1) / block.y);
    transpose_qweight_kernel<<<grid, block, 0, stream>>>(src, dst, K, packed_N);
}

// =============================================================================
// Fused Gate+Up+SwiGLU GEMV with AWQ dequant (M=1 only)
// =============================================================================
/**
 * Fuses W1 (gate) GEMV + W3 (up) GEMV + SwiGLU into a single kernel.
 * Phase 1: compute gate = W1 * x (vectorized coalesced GEMV)
 * Phase 2: compute up = W3 * x (reuses x from L2 cache)
 * Phase 3: output = SiLU(gate) * up
 *
 * Benefits: eliminates 2 kernel launches and intermediate buffer traffic.
 */
__global__ __launch_bounds__(256, 2)
void awq_fused_gate_up_swiglu_kernel(
    const half* __restrict__ X,
    const int32_t* __restrict__ w1_qweight_t,  // [N/8, K] transposed
    const int32_t* __restrict__ w1_qzeros,     // [K/G, N/8]
    const half* __restrict__ w1_scales,        // [K/G, N]
    const int32_t* __restrict__ w3_qweight_t,  // [N/8, K] transposed
    const int32_t* __restrict__ w3_qzeros,     // [K/G, N/8]
    const half* __restrict__ w3_scales,        // [K/G, N]
    half* __restrict__ Y,                      // [N]
    int K,
    int N,
    int group_size
) {
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;

    const int packed_out_idx = blockIdx.x * 8 + warp_id;
    const int out_base = packed_out_idx * 8;

    if (out_base >= N) return;

    const int packed_N = N / 8;
    const int n_groups = K / group_size;

    // Phase 1: gate = W1 * x
    float gate_acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    {
        const int32_t* warp_w1 = w1_qweight_t + packed_out_idx * K;

        for (int g = 0; g < n_groups; g++) {
            const uint32_t qz = static_cast<uint32_t>(__ldg(&w1_qzeros[g * packed_N + packed_out_idx]));
            uint32_t z_h[4];
            lop3_extract_int4_to_fp16x2(qz, z_h);

            const uint4 scale_vec = *reinterpret_cast<const uint4*>(&w1_scales[g * N + out_base]);
            const half2* s_h2 = reinterpret_cast<const half2*>(&scale_vec);

            half2 neg_sz_h2[4];
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                half2 z_h2 = *reinterpret_cast<const half2*>(&z_h[j]);
                neg_sz_h2[j] = __hneg2(__hmul2(s_h2[j], z_h2));
            }

            const int group_start = g * group_size;

            for (int k = lane_id * 4; k < group_size; k += 128) {
                const int k_idx = group_start + k;
                const uint4 w4 = *reinterpret_cast<const uint4*>(&warp_w1[k_idx]);
                const uint2 x2 = *reinterpret_cast<const uint2*>(&X[k_idx]);
                const half* x_ptr = reinterpret_cast<const half*>(&x2);
                const uint32_t w_arr[4] = {w4.x, w4.y, w4.z, w4.w};

                #pragma unroll
                for (int v = 0; v < 4; v++) {
                    const half2 x_h2 = __half2half2(x_ptr[v]);
                    uint32_t w_h[4];
                    lop3_extract_int4_to_fp16x2(w_arr[v], w_h);
                    #pragma unroll
                    for (int j = 0; j < 4; j++) {
                        half2 w_h2 = *reinterpret_cast<const half2*>(&w_h[j]);
                        half2 dq_h2 = __hfma2(s_h2[j], w_h2, neg_sz_h2[j]);
                        half2 prod = __hmul2(x_h2, dq_h2);
                        gate_acc[j * 2]     += __low2float(prod);
                        gate_acc[j * 2 + 1] += __high2float(prod);
                    }
                }
            }
        }
    }

    // Phase 2: up = W3 * x (x reads from L2 cache)
    float up_acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    {
        const int32_t* warp_w3 = w3_qweight_t + packed_out_idx * K;

        for (int g = 0; g < n_groups; g++) {
            const uint32_t qz = static_cast<uint32_t>(__ldg(&w3_qzeros[g * packed_N + packed_out_idx]));
            uint32_t z_h[4];
            lop3_extract_int4_to_fp16x2(qz, z_h);

            const uint4 scale_vec = *reinterpret_cast<const uint4*>(&w3_scales[g * N + out_base]);
            const half2* s_h2 = reinterpret_cast<const half2*>(&scale_vec);

            half2 neg_sz_h2[4];
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                half2 z_h2 = *reinterpret_cast<const half2*>(&z_h[j]);
                neg_sz_h2[j] = __hneg2(__hmul2(s_h2[j], z_h2));
            }

            const int group_start = g * group_size;

            for (int k = lane_id * 4; k < group_size; k += 128) {
                const int k_idx = group_start + k;
                const uint4 w4 = *reinterpret_cast<const uint4*>(&warp_w3[k_idx]);
                const uint2 x2 = *reinterpret_cast<const uint2*>(&X[k_idx]);
                const half* x_ptr = reinterpret_cast<const half*>(&x2);
                const uint32_t w_arr[4] = {w4.x, w4.y, w4.z, w4.w};

                #pragma unroll
                for (int v = 0; v < 4; v++) {
                    const half2 x_h2 = __half2half2(x_ptr[v]);
                    uint32_t w_h[4];
                    lop3_extract_int4_to_fp16x2(w_arr[v], w_h);
                    #pragma unroll
                    for (int j = 0; j < 4; j++) {
                        half2 w_h2 = *reinterpret_cast<const half2*>(&w_h[j]);
                        half2 dq_h2 = __hfma2(s_h2[j], w_h2, neg_sz_h2[j]);
                        half2 prod = __hmul2(x_h2, dq_h2);
                        up_acc[j * 2]     += __low2float(prod);
                        up_acc[j * 2 + 1] += __high2float(prod);
                    }
                }
            }
        }
    }

    // Warp reduction for both gate and up
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            gate_acc[i] += __shfl_down_sync(0xffffffff, gate_acc[i], offset);
            up_acc[i]   += __shfl_down_sync(0xffffffff, up_acc[i], offset);
        }
    }

    // Phase 3: SwiGLU fusion → output = SiLU(gate) * up
    if (lane_id == 0) {
        half out_half[8];
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            float gate = gate_acc[i];
            float silu_gate = gate / (1.0f + expf(-gate));
            out_half[i] = __float2half(silu_gate * up_acc[i]);
        }
        *reinterpret_cast<uint4*>(&Y[out_base]) = *reinterpret_cast<uint4*>(out_half);
    }
}

void awq_fused_gate_up_swiglu_cu(
    const half* input,
    const int32_t* w1_qweight_t,
    const int32_t* w1_qzeros,
    const half* w1_scales,
    const int32_t* w3_qweight_t,
    const int32_t* w3_qzeros,
    const half* w3_scales,
    half* output,
    int K,
    int hidden_dim,
    int group_size,
    cudaStream_t stream
) {
    const int num_blocks = (hidden_dim + 63) / 64;
    awq_fused_gate_up_swiglu_kernel<<<num_blocks, 256, 0, stream>>>(
        input, w1_qweight_t, w1_qzeros, w1_scales,
        w3_qweight_t, w3_qzeros, w3_scales,
        output, K, hidden_dim, group_size
    );
}

// =============================================================================
// Fused QKV GEMV: Q, K, V projections in a single kernel launch
// =============================================================================
/**
 * Block assignment:
 *   blocks [0, q_blocks)                           → Q projection
 *   blocks [q_blocks, q_blocks + k_blocks)         → K projection
 *   blocks [q_blocks + k_blocks, total_blocks)     → V projection
 *
 * All blocks share the same input X and use the coalesced vectorized GEMV body.
 */
__global__ __launch_bounds__(256, 4)
void awq_fused_qkv_kernel(
    const half* __restrict__ X,
    const int32_t* __restrict__ q_qwt, const int32_t* __restrict__ q_qz, const half* __restrict__ q_sc,
    half* __restrict__ q_out, int q_N,
    const int32_t* __restrict__ k_qwt, const int32_t* __restrict__ k_qz, const half* __restrict__ k_sc,
    half* __restrict__ k_out, int k_N,
    const int32_t* __restrict__ v_qwt, const int32_t* __restrict__ v_qz, const half* __restrict__ v_sc,
    half* __restrict__ v_out, int v_N,
    int K, int group_size
) {
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;

    const int q_blocks = (q_N + 63) / 64;
    const int k_blocks = (k_N + 63) / 64;

    // Determine projection and local block index
    const int32_t* qweight_t;
    const int32_t* qzeros;
    const half* scales;
    half* output;
    int N;
    int local_block;

    if (blockIdx.x < q_blocks) {
        local_block = blockIdx.x;
        qweight_t = q_qwt; qzeros = q_qz; scales = q_sc; output = q_out; N = q_N;
    } else if (blockIdx.x < q_blocks + k_blocks) {
        local_block = blockIdx.x - q_blocks;
        qweight_t = k_qwt; qzeros = k_qz; scales = k_sc; output = k_out; N = k_N;
    } else {
        local_block = blockIdx.x - q_blocks - k_blocks;
        qweight_t = v_qwt; qzeros = v_qz; scales = v_sc; output = v_out; N = v_N;
    }

    const int packed_out_idx = local_block * 8 + warp_id;
    const int out_base = packed_out_idx * 8;
    if (out_base >= N) return;

    const int packed_N = N / 8;
    const int n_groups = K / group_size;
    const int32_t* warp_qweight = qweight_t + packed_out_idx * K;

    float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};

    for (int g = 0; g < n_groups; g++) {
        const uint32_t qz = static_cast<uint32_t>(__ldg(&qzeros[g * packed_N + packed_out_idx]));
        uint32_t z_h[4];
        lop3_extract_int4_to_fp16x2(qz, z_h);

        const uint4 scale_vec = *reinterpret_cast<const uint4*>(&scales[g * N + out_base]);
        const half2* s_h2 = reinterpret_cast<const half2*>(&scale_vec);

        half2 neg_sz_h2[4];
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            half2 z_h2 = *reinterpret_cast<const half2*>(&z_h[j]);
            neg_sz_h2[j] = __hneg2(__hmul2(s_h2[j], z_h2));
        }

        const int group_start = g * group_size;

        for (int k = lane_id * 4; k < group_size; k += 128) {
            const int k_idx = group_start + k;
            const uint4 w4 = *reinterpret_cast<const uint4*>(&warp_qweight[k_idx]);
            const uint2 x2 = *reinterpret_cast<const uint2*>(&X[k_idx]);
            const half* x_ptr = reinterpret_cast<const half*>(&x2);
            const uint32_t w_arr[4] = {w4.x, w4.y, w4.z, w4.w};

            #pragma unroll
            for (int v = 0; v < 4; v++) {
                const half2 x_h2 = __half2half2(x_ptr[v]);
                uint32_t w_h[4];
                lop3_extract_int4_to_fp16x2(w_arr[v], w_h);
                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    half2 w_h2 = *reinterpret_cast<const half2*>(&w_h[j]);
                    half2 dq_h2 = __hfma2(s_h2[j], w_h2, neg_sz_h2[j]);
                    half2 prod = __hmul2(x_h2, dq_h2);
                    acc[j * 2]     += __low2float(prod);
                    acc[j * 2 + 1] += __high2float(prod);
                }
            }
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            acc[i] += __shfl_down_sync(0xffffffff, acc[i], offset);
        }
    }

    if (lane_id == 0) {
        half out_half[8];
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            out_half[i] = __float2half(acc[i]);
        }
        *reinterpret_cast<uint4*>(&output[out_base]) = *reinterpret_cast<uint4*>(out_half);
    }
}

void awq_fused_qkv_cu(
    const half* input,
    const int32_t* q_qweight_t, const int32_t* q_qzeros, const half* q_scales,
    half* q_output, int q_N,
    const int32_t* k_qweight_t, const int32_t* k_qzeros, const half* k_scales,
    half* k_output, int k_N,
    const int32_t* v_qweight_t, const int32_t* v_qzeros, const half* v_scales,
    half* v_output, int v_N,
    int K, int group_size,
    cudaStream_t stream
) {
    const int total_blocks = (q_N + 63) / 64 + (k_N + 63) / 64 + (v_N + 63) / 64;
    awq_fused_qkv_kernel<<<total_blocks, 256, 0, stream>>>(
        input,
        q_qweight_t, q_qzeros, q_scales, q_output, q_N,
        k_qweight_t, k_qzeros, k_scales, k_output, k_N,
        v_qweight_t, v_qzeros, v_scales, v_output, v_N,
        K, group_size
    );
}

}  // namespace kernel
