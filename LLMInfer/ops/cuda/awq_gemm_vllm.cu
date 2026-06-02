/**
 * AWQ GEMM Kernel with vllm-format repacked weights
 * 
 * After weight repacking, the format is:
 * vllm AWQ packing order: {0,2,4,6,1,3,5,7}
 * - bits 0:15  (lower half): elements 0,2,4,6 (even indices)
 * - bits 16:31 (upper half): elements 1,3,5,7 (odd indices)
 *
 * This layout allows using fast LOP3-based dequantization that produces
 * pairs of elements: (0,1), (2,3), (4,5), (6,7) using half2 operations.
 *
 * scales: stored linearly [scale0, scale1, ..., scale7]
 */

#include "awq_gemm_vllm.cuh"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace kernel {

// ============================================================================
// Fast LOP3-based dequantization for vllm format (after repacking)
// ============================================================================
// 
// vllm packing (after repack): bits 0-15 = even elements, bits 16-31 = odd elements
// LOP3 extracts: bits[0:3,16:19] -> (elem0, elem1)
//                bits[4:7,20:23] -> (elem2, elem3)  
//                bits[8:11,24:27] -> (elem4, elem5)
//                bits[12:15,28:31] -> (elem6, elem7)
//
// This matches the scale layout for half2 FMA operations!

__device__ __forceinline__ void dequant_vllm_lop3(
    uint32_t packed_w,
    uint32_t packed_z,
    half2* scales_h2,  // 4 x half2: (s0,s1), (s2,s3), (s4,s5), (s6,s7)
    half2* output      // 4 x half2
) {
    // LOP3 magic numbers for extracting INT4 pairs as FP16
    constexpr uint32_t FP16_TOP_MAGIC = 0x64006400;  // 1024.0h in FP16 (for subtracting)
    constexpr uint32_t BOTTOM_MASK = 0x000f000f;     // Extract bits 0:3 and 16:19
    constexpr uint32_t TOP_MASK = 0x00f000f0;        // Extract bits 4:7 and 20:23
    constexpr uint32_t I4s_TO_FP16_MAGIC = 0x64006400;  // Add to convert to FP16

    uint32_t w_tmp1, w_tmp2, z_tmp1, z_tmp2;

    // Process lower 8 bits (4:7, 0:3) from both halves simultaneously
    // Extract pair 0 (bits 0:3 and 16:19) -> elements 0,1
    asm volatile("lop3.b32 %0, %1, %2, %3, 0xea;" : "=r"(w_tmp1) : "r"(packed_w), "n"(BOTTOM_MASK), "n"(I4s_TO_FP16_MAGIC));
    asm volatile("lop3.b32 %0, %1, %2, %3, 0xea;" : "=r"(z_tmp1) : "r"(packed_z), "n"(BOTTOM_MASK), "n"(I4s_TO_FP16_MAGIC));
    
    // Extract pair 1 (bits 4:7 and 20:23) -> elements 2,3, shift right 4
    asm volatile("lop3.b32 %0, %1, %2, %3, 0xea;" : "=r"(w_tmp2) : "r"(packed_w), "n"(TOP_MASK), "n"(I4s_TO_FP16_MAGIC));
    asm volatile("lop3.b32 %0, %1, %2, %3, 0xea;" : "=r"(z_tmp2) : "r"(packed_z), "n"(TOP_MASK), "n"(I4s_TO_FP16_MAGIC));
    
    // Convert to FP16 by subtracting magic (1024.0)
    half2 w01 = __hsub2(*reinterpret_cast<half2*>(&w_tmp1), *reinterpret_cast<const half2*>(&FP16_TOP_MAGIC));
    half2 z01 = __hsub2(*reinterpret_cast<half2*>(&z_tmp1), *reinterpret_cast<const half2*>(&FP16_TOP_MAGIC));
    half2 w23 = __hsub2(*reinterpret_cast<half2*>(&w_tmp2), *reinterpret_cast<const half2*>(&FP16_TOP_MAGIC));
    half2 z23 = __hsub2(*reinterpret_cast<half2*>(&z_tmp2), *reinterpret_cast<const half2*>(&FP16_TOP_MAGIC));
    w23 = __hmul2(w23, __float2half2_rn(0.0625f));  // Divide by 16 (shift right 4)
    z23 = __hmul2(z23, __float2half2_rn(0.0625f));
    
    // Process upper 8 bits (12:15, 8:11) by shifting right 8
    uint32_t packed_w_hi = packed_w >> 8;
    uint32_t packed_z_hi = packed_z >> 8;
    
    asm volatile("lop3.b32 %0, %1, %2, %3, 0xea;" : "=r"(w_tmp1) : "r"(packed_w_hi), "n"(BOTTOM_MASK), "n"(I4s_TO_FP16_MAGIC));
    asm volatile("lop3.b32 %0, %1, %2, %3, 0xea;" : "=r"(z_tmp1) : "r"(packed_z_hi), "n"(BOTTOM_MASK), "n"(I4s_TO_FP16_MAGIC));
    asm volatile("lop3.b32 %0, %1, %2, %3, 0xea;" : "=r"(w_tmp2) : "r"(packed_w_hi), "n"(TOP_MASK), "n"(I4s_TO_FP16_MAGIC));
    asm volatile("lop3.b32 %0, %1, %2, %3, 0xea;" : "=r"(z_tmp2) : "r"(packed_z_hi), "n"(TOP_MASK), "n"(I4s_TO_FP16_MAGIC));

    half2 w45 = __hsub2(*reinterpret_cast<half2*>(&w_tmp1), *reinterpret_cast<const half2*>(&FP16_TOP_MAGIC));
    half2 z45 = __hsub2(*reinterpret_cast<half2*>(&z_tmp1), *reinterpret_cast<const half2*>(&FP16_TOP_MAGIC));
    half2 w67 = __hsub2(*reinterpret_cast<half2*>(&w_tmp2), *reinterpret_cast<const half2*>(&FP16_TOP_MAGIC));
    half2 z67 = __hsub2(*reinterpret_cast<half2*>(&z_tmp2), *reinterpret_cast<const half2*>(&FP16_TOP_MAGIC));
    w67 = __hmul2(w67, __float2half2_rn(0.0625f));
    z67 = __hmul2(z67, __float2half2_rn(0.0625f));

    // Apply dequant: output = scale * (w - z)
    output[0] = __hmul2(scales_h2[0], __hsub2(w01, z01));  // (o0, o1)
    output[1] = __hmul2(scales_h2[1], __hsub2(w23, z23));  // (o2, o3)
    output[2] = __hmul2(scales_h2[2], __hsub2(w45, z45));  // (o4, o5)
    output[3] = __hmul2(scales_h2[3], __hsub2(w67, z67));  // (o6, o7)
}

// Simple scalar dequant matching awq_gemm_fast.cu logic (known to work)
__device__ __forceinline__ void dequant_vllm_scalar(
    uint32_t packed_w,
    uint32_t packed_z,
    const half* scales_ptr,
    half* output
) {
    // Use the same formula as dequant_awq_int4_to_fp16 (verified working)
    // AWQ bit positions: output index i -> bit position
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int bit_pos = ((i & 1) == 0) ? (i / 2) : (4 + i / 2);
        bit_pos *= 4;
        
        int w = (packed_w >> bit_pos) & 0xF;
        int z = (packed_z >> bit_pos) & 0xF;
        float scale = __half2float(scales_ptr[i]);
        output[i] = __float2half(scale * (float)(w - z));
    }
}

template <int N>
__global__ void __launch_bounds__(64)
awq_gemm_vllm_kernel(
    int G, half* __restrict__ A, int* __restrict__ B,
    half* __restrict__ scaling_factors, int* __restrict__ zeros,
    int M, int IC, int OC, half* __restrict__ C
) {
    static_assert(N == 64 || N == 128, "N must be 64 or 128");

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 750
    return;
#else
    float C_warp[32];
    __shared__ half A_shared[16 * (32 + 8)];
    __shared__ half B_shared[32 * (N + 8)];

    int j_factors1 = ((OC + N - 1) / N);
    int blockIdx_y = blockIdx.x % ((M + 16 - 1) / 16 * j_factors1);

    half A_shared_warp[8];
    half B_shared_warp[N / 4];
    
    for (int j = 0; j < N / 32; ++j) {
        for (int i = 0; i < 8; ++i) C_warp[(j * 8) + i] = 0.0f;
    }

    static constexpr int row_stride_warp = 32 * 8 / 32;
    static constexpr int row_stride = 2 * 32 * 8 / N;
    
    bool ld_A_flag = (blockIdx_y / j_factors1 * 16 + threadIdx.y * row_stride_warp +
                      threadIdx.x * 8 / 32) < M;

    half* A_ptr = A + (blockIdx_y / j_factors1 * 16 + threadIdx.y * row_stride_warp + threadIdx.x / 4) * IC +
                  (threadIdx.x % 4) * 8;

    int* B_ptr = B + threadIdx.y * (OC / 8) * (256 / N) +
                 (threadIdx.x / (N / 8)) * (OC / 8) +
                 (blockIdx_y % j_factors1) * (N / 8) +
                 (threadIdx.x % (N / 8));

    half* A_shared_ptr = A_shared + threadIdx.y * row_stride_warp * (32 + 8) +
                         (threadIdx.x / 4) * (32 + 8) + (threadIdx.x % 4) * 8;

    half* B_shared_ptr = B_shared + threadIdx.y * (row_stride / 2) * (N + 8) +
                         (threadIdx.x / (N/8)) * (N + 8) + (threadIdx.x % (N/8)) * 8;

    int* zeros_ptr = zeros + (blockIdx_y % j_factors1) * (N / 8) + (threadIdx.x % (N/8));
    half* sf_ptr = scaling_factors + (blockIdx_y % j_factors1) * N + (threadIdx.x % (N/8)) * 8;
    half* C_ptr = C + (blockIdx_y % j_factors1) * N + threadIdx.y * (N / 2) + (threadIdx.x % 4) * 2;

    int k_bound = IC / 32;
    
    for (int k_0_0 = 0; k_0_0 < k_bound; ++k_0_0) {
        __syncthreads();
        
        if (ld_A_flag) *(uint4*)(A_shared_ptr) = *(uint4*)(A_ptr + k_0_0 * 32);
        else *(uint4*)(A_shared_ptr) = make_uint4(0, 0, 0, 0);

        // Load zeros and scales for this K group
        uint32_t zeros_loaded = *(uint32_t*)(zeros_ptr + k_0_0 * 32 / G * (OC / 8));
        half* scales_loaded = sf_ptr + k_0_0 * 32 / G * OC;

        int* B_ptr_local = B_ptr + k_0_0 * 32 * (OC / 8);

        for (int ax0 = 0; ax0 < N / 16; ++ax0) {
            // Load weight
            uint32_t B_loaded = *(uint32_t*)(B_ptr_local + ax0 * row_stride * (OC / 8));
            
            // Dequantize with LOP3 for vllm format (fast path)
            half2 B_dequant_h2[4];
            half2* scales_h2 = reinterpret_cast<half2*>(scales_loaded);
            dequant_vllm_lop3(B_loaded, zeros_loaded, scales_h2, B_dequant_h2);

            *(uint4*)(B_shared_ptr + ax0 * row_stride * (N + 8)) = *(uint4*)B_dequant_h2;
        }
        
        __syncthreads();

        for (int k_0_1 = 0; k_0_1 < 2; ++k_0_1) {
            unsigned int addr;
            asm volatile(
                "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
                : "=r"(addr)
                : "l"((void*)((&(A_shared[(k_0_1 * 16)])) + ((threadIdx.x & 15) * 40) + ((threadIdx.x >> 4) * 8))));
            asm volatile(
                "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(((unsigned*)(A_shared_warp))[0]), "=r"(((unsigned*)(A_shared_warp))[1]),
                  "=r"(((unsigned*)(A_shared_warp))[2]), "=r"(((unsigned*)(A_shared_warp))[3])
                : "r"(addr));

            for (int ax1_0 = 0; ax1_0 < N / 32; ++ax1_0) {
                asm volatile(
                    "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
                    : "=r"(addr)
                    : "l"((void*)((&(B_shared[(k_0_1 * (N * 16 + 128)) + threadIdx.y * (N / 2) + ax1_0 * 16])) +
                          ((threadIdx.x & 15) * (N + 8)) + ((threadIdx.x >> 4) * 8))));
                asm volatile(
                    "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(((unsigned*)(B_shared_warp + ax1_0 * 8))[0]), "=r"(((unsigned*)(B_shared_warp + ax1_0 * 8))[1]),
                      "=r"(((unsigned*)(B_shared_warp + ax1_0 * 8))[2]), "=r"(((unsigned*)(B_shared_warp + ax1_0 * 8))[3])
                    : "r"(addr));
            }

            for (int j_0_4 = 0; j_0_4 < N / 32; ++j_0_4) {
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                    : "=f"(((float*)(C_warp + j_0_4 * 8))[0]), "=f"(((float*)(C_warp + j_0_4 * 8))[1]),
                      "=f"(((float*)(C_warp + j_0_4 * 8))[2]), "=f"(((float*)(C_warp + j_0_4 * 8))[3])
                    : "r"(((unsigned*)(A_shared_warp))[0]), "r"(((unsigned*)(A_shared_warp))[1]),
                      "r"(((unsigned*)(A_shared_warp))[2]), "r"(((unsigned*)(A_shared_warp))[3]),
                      "r"(((unsigned*)(B_shared_warp + j_0_4 * 8))[0]), "r"(((unsigned*)(B_shared_warp + j_0_4 * 8))[1]),
                      "f"(((float*)(C_warp + j_0_4 * 8))[0]), "f"(((float*)(C_warp + j_0_4 * 8))[1]),
                      "f"(((float*)(C_warp + j_0_4 * 8))[2]), "f"(((float*)(C_warp + j_0_4 * 8))[3]));
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                    : "=f"(((float*)(C_warp + j_0_4 * 8 + 4))[0]), "=f"(((float*)(C_warp + j_0_4 * 8 + 4))[1]),
                      "=f"(((float*)(C_warp + j_0_4 * 8 + 4))[2]), "=f"(((float*)(C_warp + j_0_4 * 8 + 4))[3])
                    : "r"(((unsigned*)(A_shared_warp))[0]), "r"(((unsigned*)(A_shared_warp))[1]),
                      "r"(((unsigned*)(A_shared_warp))[2]), "r"(((unsigned*)(A_shared_warp))[3]),
                      "r"(((unsigned*)(B_shared_warp + j_0_4 * 8 + 4))[0]), "r"(((unsigned*)(B_shared_warp + j_0_4 * 8 + 4))[1]),
                      "f"(((float*)(C_warp + j_0_4 * 8 + 4))[0]), "f"(((float*)(C_warp + j_0_4 * 8 + 4))[1]),
                      "f"(((float*)(C_warp + j_0_4 * 8 + 4))[2]), "f"(((float*)(C_warp + j_0_4 * 8 + 4))[3]));
            }
        }
    }

    for (int ax1 = 0; ax1 < N / 32; ++ax1) {
        for (int local_id = 0; local_id < 8; ++local_id) {
            int row_offset = (blockIdx_y / j_factors1) * 16 + (threadIdx.x / 4) + (local_id % 4) / 2 * 8;
            if (row_offset < M) {
                *(C_ptr + ax1 * 16 + row_offset * OC + (local_id / 4) * 8 + local_id % 2) =
                    __float2half(C_warp[ax1 * 8 + local_id]);
            }
        }
    }
#endif
}

void awq_gemm_vllm_cu(
    const half* input, const int32_t* qweight, const int32_t* qzeros, const half* scales,
    half* output, int M, int K, int N, int group_size, cudaStream_t stream
) {
    if (N >= 128 && N % 128 == 0) {
        int j_factors1 = N / 128;
        dim3 num_blocks((M + 16 - 1) / 16 * j_factors1);
        dim3 threads_per_block(32, 2);
        awq_gemm_vllm_kernel<128><<<num_blocks, threads_per_block, 0, stream>>>(
            group_size, const_cast<half*>(input), const_cast<int*>(qweight),
            const_cast<half*>(scales), const_cast<int*>(qzeros), M, K, N, output);
    } else if (N % 64 == 0) {
        int j_factors1 = N / 64;
        dim3 num_blocks((M + 16 - 1) / 16 * j_factors1);
        dim3 threads_per_block(32, 2);
        awq_gemm_vllm_kernel<64><<<num_blocks, threads_per_block, 0, stream>>>(
            group_size, const_cast<half*>(input), const_cast<int*>(qweight),
            const_cast<half*>(scales), const_cast<int*>(qzeros), M, K, N, output);
    }
}

}  // namespace kernel
