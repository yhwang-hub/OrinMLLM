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
// GEMV Kernel with LOP3 Dequant for Decode (M=1)
// =============================================================================
/**
 * Memory-bandwidth optimized GEMV using LOP3 INT4 dequantization.
 *
 * Design:
 * - 256 threads/block = 8 warps, each warp handles 8 output channels
 * - Each block processes 64 output channels total
 * - LOP3 extracts INT4 weights as half2 pairs (4 LOP3 + 4 convert ops
 *   instead of 8 scalar shift+mask+cast operations)
 * - half2 FMA for paired dequant+accumulate
 * - Float accumulators for numerical stability across groups
 * - Warp shuffle for final K-dimension reduction
 */
__global__ __launch_bounds__(256, 4)
void awq_gemv_fast_kernel(
    const half* __restrict__ X,           // [K]
    const int32_t* __restrict__ qweight,  // [K, N/8]
    const int32_t* __restrict__ qzeros,   // [K/G, N/8]
    const half* __restrict__ scales,      // [K/G, N]
    half* __restrict__ Y,                 // [N]
    int K,
    int N,
    int group_size
) {
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;

    // Each warp handles 8 output channels (one packed INT32 column)
    const int packed_out_idx = blockIdx.x * 8 + warp_id;
    const int out_base = packed_out_idx * 8;

    if (out_base >= N) return;

    const int packed_N = N / 8;
    const int n_groups = K / group_size;

    // FP32 accumulators for 8 outputs
    float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};

    for (int g = 0; g < n_groups; g++) {
        // --- Per-group setup (amortized over group_size K iterations) ---

        // LOP3 extract zeros → 4 half2 pairs: (z0,z1), (z2,z3), (z4,z5), (z6,z7)
        const uint32_t qz = static_cast<uint32_t>(__ldg(&qzeros[g * packed_N + packed_out_idx]));
        uint32_t z_h[4];
        lop3_extract_int4_to_fp16x2(qz, z_h);

        // Load scales as 4 half2 pairs
        const uint4 scale_vec = *reinterpret_cast<const uint4*>(&scales[g * N + out_base]);
        const half2* s_h2 = reinterpret_cast<const half2*>(&scale_vec);

        // Precompute neg_scale_zero = -(scale * zero) for FMA:
        //   scale * w + neg_sz = scale * (w - zero)
        half2 neg_sz_h2[4];
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            half2 z_h2 = *reinterpret_cast<const half2*>(&z_h[j]);
            neg_sz_h2[j] = __hneg2(__hmul2(s_h2[j], z_h2));
        }

        const int group_start = g * group_size;

        // --- Inner loop over K dimension ---
        for (int k = lane_id; k < group_size; k += 32) {
            const int k_idx = group_start + k;

            // Load input and broadcast to half2
            const half x_val = __ldg(&X[k_idx]);
            const half2 x_h2 = __half2half2(x_val);

            // Load packed weight and extract with LOP3
            const uint32_t w_packed = static_cast<uint32_t>(__ldg(&qweight[k_idx * packed_N + packed_out_idx]));
            uint32_t w_h[4];
            lop3_extract_int4_to_fp16x2(w_packed, w_h);

            // Dequant and accumulate: acc += x * scale * (w - zero)
            //   = x * (scale * w + neg_sz)
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                half2 w_h2 = *reinterpret_cast<const half2*>(&w_h[j]);
                // dequant = scale * w - scale * zero = scale * (w - zero)
                half2 dq_h2 = __hfma2(s_h2[j], w_h2, neg_sz_h2[j]);
                half2 prod = __hmul2(x_h2, dq_h2);
                acc[j * 2]     += __low2float(prod);
                acc[j * 2 + 1] += __high2float(prod);
            }
        }
    }

    // Warp reduction using shuffle
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            acc[i] += __shfl_down_sync(0xffffffff, acc[i], offset);
        }
    }

    // Write output (only lane 0 of each warp)
    if (lane_id == 0) {
        half out_half[8];
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            out_half[i] = __float2half(acc[i]);
        }
        *reinterpret_cast<uint4*>(&Y[out_base]) = *reinterpret_cast<uint4*>(out_half);
    }
}

// =============================================================================
// Dispatcher (M=1 only — called from awq_gemm_tensorcore_cu)
// =============================================================================

void awq_gemm_fast_cu(
    const half* input,
    const int32_t* qweight,
    const int32_t* qzeros,
    const half* scales,
    half* output,
    int M,
    int K,
    int N,
    int group_size,
    cudaStream_t stream
) {
    // This function is only called when M=1 from awq_gemm_tensorcore_cu.
    // GEMV: memory-bandwidth optimized with LOP3 dequantization.
    // Each block handles 64 output channels (8 warps × 8 outputs/warp).
    const int num_blocks = (N + 63) / 64;
    awq_gemv_fast_kernel<<<num_blocks, 256, 0, stream>>>(
        input, qweight, qzeros, scales, output,
        K, N, group_size
    );
}

}  // namespace kernel
