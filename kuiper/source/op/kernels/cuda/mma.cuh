#pragma once
/**
 * MMA (Matrix Multiply Accumulate) primitives using PTX instructions
 * Optimized for NVIDIA Orin (SM 8.7, Ampere-based architecture)
 * 
 * This file provides tensor core operations similar to nvcuda::wmma but with
 * better control over memory layout and performance.
 * 
 * References:
 * - NVIDIA PTX ISA: https://docs.nvidia.com/cuda/parallel-thread-execution/
 * - llama.cpp mma.cuh implementation
 */

#include <cuda_fp16.h>
#include <mma.h>

namespace kuiper_mma {

// Warp size constant
constexpr int WARP_SIZE = 32;

// Check for Ampere MMA support (SM >= 8.0)
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#define AMPERE_MMA_AVAILABLE
#endif

// Check for Turing MMA support (SM >= 7.5)  
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
#define TURING_MMA_AVAILABLE
#endif

// ============================================================================
// MMA Tile types - matching PTX mma.m16n8k16 and mma.m16n8k8 shapes
// ============================================================================

/**
 * Tile structure for MMA operations
 * I: number of rows
 * J: number of columns (in 32-bit elements)
 * T: element type
 */
template <int I_, int J_, typename T>
struct tile {
    static constexpr int I = I_;
    static constexpr int J = J_;
    static constexpr int ne = (I * J) / WARP_SIZE;  // Elements per thread
    T x[ne] = {0};
    
    __device__ __forceinline__ void fill(T val) {
        #pragma unroll
        for (int i = 0; i < ne; ++i) {
            x[i] = val;
        }
    }
};

// Common tile types for FP16 MMA on Ampere
// m16n8k16: A is 16x16, B is 16x8, C is 16x8 
using tile_A_fp16 = tile<16, 16, half2>;   // A matrix tile (row-major)
using tile_B_fp16 = tile<16, 8, half2>;    // B matrix tile (col-major) 
using tile_C_fp32 = tile<16, 8, float>;    // C accumulator tile

// Smaller tiles for different configurations
using tile_A_8x8 = tile<8, 8, half2>;
using tile_B_8x8 = tile<8, 8, half2>;
using tile_C_8x8 = tile<8, 8, float>;

// ============================================================================
// LDMATRIX - Load matrix tiles from shared memory
// ============================================================================

/**
 * Load 16x16 half2 tile from shared memory using ldmatrix
 * ptr must be aligned to 16 bytes and point to shared memory
 */
template<int stride>
__device__ __forceinline__ void load_ldmatrix_16x16(
    tile<16, 16, half2>& tile_out,
    const half2* ptr
) {
#ifdef AMPERE_MMA_AVAILABLE
    // ldmatrix.sync.aligned.m8n8.x4.shared.b16
    // Loads 4x 8x8 matrices = 16x16 tile
    const unsigned int ptr_shared = __cvta_generic_to_shared(ptr);
    const int lane_id = threadIdx.x % WARP_SIZE;
    
    // Calculate row offset for this thread
    const int row = lane_id % 16;
    const int col_group = lane_id / 16;
    
    // Load using ldmatrix instruction
    unsigned int* out = reinterpret_cast<unsigned int*>(tile_out.x);
    
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(out[0]), "=r"(out[1]), "=r"(out[2]), "=r"(out[3])
        : "r"(ptr_shared + (row * stride + col_group * 8) * sizeof(half2))
    );
#else
    // Fallback: manual load
    const int lane_id = threadIdx.x % WARP_SIZE;
    #pragma unroll
    for (int i = 0; i < tile_out.ne; ++i) {
        const int row = (lane_id + i * WARP_SIZE / tile_out.ne) % 16;
        const int col = (lane_id / 16 + i) % 8;
        tile_out.x[i] = ptr[row * stride + col];
    }
#endif
}

/**
 * Load 16x8 half2 tile from shared memory
 */
template<int stride>
__device__ __forceinline__ void load_ldmatrix_16x8(
    tile<16, 8, half2>& tile_out,
    const half2* ptr
) {
#ifdef AMPERE_MMA_AVAILABLE
    const unsigned int ptr_shared = __cvta_generic_to_shared(ptr);
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int row = lane_id % 16;
    
    unsigned int* out = reinterpret_cast<unsigned int*>(tile_out.x);
    
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%4];"
        : "=r"(out[0]), "=r"(out[1])
        : "r"(ptr_shared + row * stride * sizeof(half2))
    );
#else
    const int lane_id = threadIdx.x % WARP_SIZE;
    #pragma unroll
    for (int i = 0; i < tile_out.ne; ++i) {
        const int row = (lane_id + i * 8) % 16;
        const int col = (lane_id / 16 + i) % 4;
        tile_out.x[i] = ptr[row * stride + col];
    }
#endif
}

// ============================================================================
// MMA Instructions - FP16 Matrix Multiply
// ============================================================================

/**
 * FP16 matrix multiply-accumulate: C = A @ B + C
 * Uses mma.m16n8k16.row.col.f32.f16.f16.f32 instruction
 * 
 * A: 16x16 half matrix (row-major in shared memory)
 * B: 16x8 half matrix (col-major in shared memory)  
 * C: 16x8 float accumulator
 */
__device__ __forceinline__ void mma_fp16_m16n8k16(
    tile_C_fp32& C,
    const tile_A_fp16& A,
    const tile_B_fp16& B
) {
#ifdef AMPERE_MMA_AVAILABLE
    const unsigned int* a = reinterpret_cast<const unsigned int*>(A.x);
    const unsigned int* b = reinterpret_cast<const unsigned int*>(B.x);
    float* c = C.x;
    
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13};"
        : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3])
    );
#else
    // Fallback: use nvcuda::wmma
    using namespace nvcuda::wmma;
    fragment<matrix_a, 16, 16, 16, half, row_major> a_frag;
    fragment<matrix_b, 16, 16, 16, half, col_major> b_frag;
    fragment<accumulator, 16, 16, 16, float> c_frag;
    
    // Copy data to fragments and perform mma
    #pragma unroll
    for (int i = 0; i < a_frag.num_elements; ++i) {
        a_frag.x[i] = reinterpret_cast<const half*>(A.x)[i];
    }
    #pragma unroll
    for (int i = 0; i < b_frag.num_elements; ++i) {
        b_frag.x[i] = reinterpret_cast<const half*>(B.x)[i];
    }
    #pragma unroll
    for (int i = 0; i < c_frag.num_elements; ++i) {
        c_frag.x[i] = C.x[i];
    }
    
    mma_sync(c_frag, a_frag, b_frag, c_frag);
    
    #pragma unroll
    for (int i = 0; i < c_frag.num_elements; ++i) {
        C.x[i] = c_frag.x[i];
    }
#endif
}

/**
 * FP16 MMA with smaller tile: m16n8k8
 * More flexible for different matrix sizes
 */
__device__ __forceinline__ void mma_fp16_m16n8k8(
    tile<16, 8, float>& C,
    const tile<16, 8, half2>& A,
    const tile<8, 8, half2>& B
) {
#ifdef AMPERE_MMA_AVAILABLE
    const unsigned int* a = reinterpret_cast<const unsigned int*>(A.x);
    const unsigned int* b = reinterpret_cast<const unsigned int*>(B.x);
    float* c = C.x;
    
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5}, "
        "{%6}, "
        "{%7, %8, %9, %10};"
        : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
        : "r"(a[0]), "r"(a[1]),
          "r"(b[0]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3])
    );
#endif
}

// ============================================================================
// WMMA Wrapper - Higher level interface using nvcuda::wmma
// ============================================================================

/**
 * WMMA-based FP16 matrix multiply: C = A @ B
 * Uses 16x16x16 tiles
 */
template<int M, int N, int K>
__device__ __forceinline__ void wmma_gemm_fp16(
    float* C,
    const half* A,
    const half* B,
    const int lda,
    const int ldb,
    const int ldc
) {
    using namespace nvcuda::wmma;
    
    fragment<matrix_a, M, N, K, half, row_major> a_frag;
    fragment<matrix_b, M, N, K, half, col_major> b_frag;
    fragment<accumulator, M, N, K, float> c_frag;
    
    // Initialize accumulator
    fill_fragment(c_frag, 0.0f);
    
    // Load matrices
    load_matrix_sync(a_frag, A, lda);
    load_matrix_sync(b_frag, B, ldb);
    
    // Perform matrix multiply
    mma_sync(c_frag, a_frag, b_frag, c_frag);
    
    // Store result
    store_matrix_sync(C, c_frag, ldc, mem_row_major);
}

// ============================================================================
// Utility functions for MMA operations
// ============================================================================

/**
 * Warp-level reduction for softmax
 */
__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

/**
 * Half2 vector operations for better memory bandwidth
 */
__device__ __forceinline__ half2 half2_max(half2 a, half2 b) {
    return __hmax2(a, b);
}

__device__ __forceinline__ half2 half2_exp(half2 x) {
    return h2exp(x);
}

__device__ __forceinline__ float2 half2_to_float2(half2 x) {
    return __half22float2(x);
}

__device__ __forceinline__ half2 float2_to_half2(float2 x) {
    return __float22half2_rn(x);
}

/**
 * Vectorized load/store for FP16
 */
__device__ __forceinline__ void load_half4(half* dst, const half* src) {
    *reinterpret_cast<float2*>(dst) = *reinterpret_cast<const float2*>(src);
}

__device__ __forceinline__ void store_half4(half* dst, const half* src) {
    *reinterpret_cast<float2*>(dst) = *reinterpret_cast<const float2*>(src);
}

__device__ __forceinline__ void load_half8(half* dst, const half* src) {
    *reinterpret_cast<float4*>(dst) = *reinterpret_cast<const float4*>(src);
}

__device__ __forceinline__ void store_half8(half* dst, const half* src) {
    *reinterpret_cast<float4*>(dst) = *reinterpret_cast<const float4*>(src);
}

// ============================================================================
// Online Softmax utilities for Flash Attention
// ============================================================================

/**
 * Online softmax state for streaming computation
 */
struct OnlineSoftmax {
    float m;  // Running max
    float l;  // Running sum of exp(x - m)
    
    __device__ __forceinline__ OnlineSoftmax() : m(-INFINITY), l(0.0f) {}
    
    __device__ __forceinline__ void update(float x) {
        float m_new = fmaxf(m, x);
        float exp_diff = expf(m - m_new);
        l = l * exp_diff + expf(x - m_new);
        m = m_new;
    }
    
    __device__ __forceinline__ void merge(const OnlineSoftmax& other) {
        float m_new = fmaxf(m, other.m);
        l = l * expf(m - m_new) + other.l * expf(other.m - m_new);
        m = m_new;
    }
    
    __device__ __forceinline__ float normalize(float x) const {
        return expf(x - m) / l;
    }
};

/**
 * Warp-level online softmax reduction
 */
__device__ __forceinline__ OnlineSoftmax warp_reduce_softmax(OnlineSoftmax local) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        OnlineSoftmax other;
        other.m = __shfl_xor_sync(0xffffffff, local.m, offset);
        other.l = __shfl_xor_sync(0xffffffff, local.l, offset);
        local.merge(other);
    }
    return local;
}

}  // namespace kuiper_mma
