/*
 * AWQ Optimized GEMM/GEMV for KuiperLLama - Version 6.0 (WMMA-Style)
 * 
 * High-performance INT4 quantized GEMM/GEMV kernels
 * 
 * Version 6.0 Improvements:
 * 1. WMMA-based W4A16 GEMM kernel for prefill (uses Tensor Cores)
 * 2. On-the-fly dequantization in shared memory
 * 3. Large 64x64 output tiles per block
 * 4. Ultra-fast fused GEMV for decode (memory-bandwidth optimized)
 * 5. Correct AWQ reverse order unpacking: {0, 4, 1, 5, 2, 6, 3, 7}
 * 
 * Target: Outperform FP16 in BOTH prefill and decode phases
 * WITHOUT increasing memory usage (no pre-dequantization)
 * 
 * AWQ uses a specific bit packing order that is NOT sequential.
 * For 8 INT4 values in one INT32:
 *   Output index:  0  1  2  3  4  5  6  7
 *   Bit position:  0  4  8  12 16 20 24 28  (i * 4)
 *   AWQ mapping:   0  4  1  5  2  6  3  7   (reverse order)
 */

#include "awq_gemm_tensorcore.cuh"
#include "awq_gemm_v3.cuh"
#include "awq_gemm_fast.cuh"
#include "awq_gemm_vllm.cuh"
#include <cuda_fp16.h>

namespace kernel {

// AWQ reverse order lookup table (compile-time constant)
// Maps output index to bit position: bit_pos = AWQ_ORDER[i] * 4
__constant__ int AWQ_ORDER[8] = {0, 4, 1, 5, 2, 6, 3, 7};  // elem->pos mapping for vllm format

// ============================================================================
// GEMV Kernel (Decode Phase, M=1) - Highly Optimized V2
// ============================================================================

/**
 * High-performance AWQ GEMV kernel with aggressive vectorization
 * 
 * Key optimizations:
 * - Each warp computes 8 output channels (one packed int32 of weights)
 * - 8 warps per block = 64 output channels per block
 * - Vectorized scale loading (uint4)
 * - __ldg() for read-only cache (faster than L1)
 * - Correct AWQ reverse order unpacking
 * - Precompute -scale * zero outside inner loop for FMA optimization
 */
__global__ __launch_bounds__(256, 4)
void awq_gemv_kernel(
    const half* __restrict__ X,           // [in_features]
    const int32_t* __restrict__ qweight,  // [in_features, out_features/8]
    const int32_t* __restrict__ qzeros,   // [n_groups, out_features/8]
    const half* __restrict__ scales,      // [n_groups, out_features]
    half* __restrict__ Y,                 // [out_features]
    int in_features,
    int out_features,
    int group_size
) {
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    const int num_warps = blockDim.x / 32;  // 8
    
    // Each warp handles 8 output channels (one packed int32)
    const int packed_out_idx = blockIdx.x * num_warps + warp_id;
    const int out_base = packed_out_idx * 8;
    
    if (out_base >= out_features) return;
    
    const int packed_out_dim = out_features / 8;
    const int n_groups = in_features / group_size;
    
    // AWQ reverse order for unpacking
    const int awq_order[8] = {0, 4, 1, 5, 2, 6, 3, 7};  // elem->pos mapping for vllm format
    
    // Accumulators for 8 outputs (FP32 for precision)
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    float acc4 = 0.0f, acc5 = 0.0f, acc6 = 0.0f, acc7 = 0.0f;
    
    // Process all groups
    for (int g = 0; g < n_groups; g++) {
        // Load zeros for this group
        const int32_t qz_packed = __ldg(&qzeros[g * packed_out_dim + packed_out_idx]);
        
        // Load all 8 scales at once using vectorized load
        uint4 scale_vec = *reinterpret_cast<const uint4*>(&scales[g * out_features + out_base]);
        float s0 = __half2float(((half*)&scale_vec)[0]);
        float s1 = __half2float(((half*)&scale_vec)[1]);
        float s2 = __half2float(((half*)&scale_vec)[2]);
        float s3 = __half2float(((half*)&scale_vec)[3]);
        float s4 = __half2float(((half*)&scale_vec)[4]);
        float s5 = __half2float(((half*)&scale_vec)[5]);
        float s6 = __half2float(((half*)&scale_vec)[6]);
        float s7 = __half2float(((half*)&scale_vec)[7]);
        
        // Extract zeros with AWQ order and precompute -scale * zero
        float nsz0 = -s0 * (float)((qz_packed >> (awq_order[0] * 4)) & 0xF);
        float nsz1 = -s1 * (float)((qz_packed >> (awq_order[1] * 4)) & 0xF);
        float nsz2 = -s2 * (float)((qz_packed >> (awq_order[2] * 4)) & 0xF);
        float nsz3 = -s3 * (float)((qz_packed >> (awq_order[3] * 4)) & 0xF);
        float nsz4 = -s4 * (float)((qz_packed >> (awq_order[4] * 4)) & 0xF);
        float nsz5 = -s5 * (float)((qz_packed >> (awq_order[5] * 4)) & 0xF);
        float nsz6 = -s6 * (float)((qz_packed >> (awq_order[6] * 4)) & 0xF);
        float nsz7 = -s7 * (float)((qz_packed >> (awq_order[7] * 4)) & 0xF);
        
        // Process input features in this group
        const int group_start = g * group_size;
        for (int k = lane_id; k < group_size; k += 32) {
            const int in_idx = group_start + k;
            
            float x = __half2float(__ldg(&X[in_idx]));
            const int32_t w_packed = __ldg(&qweight[in_idx * packed_out_dim + packed_out_idx]);
            
            // Dequantize and accumulate with FMA
            float w0 = (float)((w_packed >> (awq_order[0] * 4)) & 0xF);
            float w1 = (float)((w_packed >> (awq_order[1] * 4)) & 0xF);
            float w2 = (float)((w_packed >> (awq_order[2] * 4)) & 0xF);
            float w3 = (float)((w_packed >> (awq_order[3] * 4)) & 0xF);
            float w4 = (float)((w_packed >> (awq_order[4] * 4)) & 0xF);
            float w5 = (float)((w_packed >> (awq_order[5] * 4)) & 0xF);
            float w6 = (float)((w_packed >> (awq_order[6] * 4)) & 0xF);
            float w7 = (float)((w_packed >> (awq_order[7] * 4)) & 0xF);
            
            acc0 = fmaf(x * s0, w0, acc0 + x * nsz0);
            acc1 = fmaf(x * s1, w1, acc1 + x * nsz1);
            acc2 = fmaf(x * s2, w2, acc2 + x * nsz2);
            acc3 = fmaf(x * s3, w3, acc3 + x * nsz3);
            acc4 = fmaf(x * s4, w4, acc4 + x * nsz4);
            acc5 = fmaf(x * s5, w5, acc5 + x * nsz5);
            acc6 = fmaf(x * s6, w6, acc6 + x * nsz6);
            acc7 = fmaf(x * s7, w7, acc7 + x * nsz7);
        }
    }
    
    // Warp reduction using shuffle
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        acc0 += __shfl_down_sync(0xffffffff, acc0, offset);
        acc1 += __shfl_down_sync(0xffffffff, acc1, offset);
        acc2 += __shfl_down_sync(0xffffffff, acc2, offset);
        acc3 += __shfl_down_sync(0xffffffff, acc3, offset);
        acc4 += __shfl_down_sync(0xffffffff, acc4, offset);
        acc5 += __shfl_down_sync(0xffffffff, acc5, offset);
        acc6 += __shfl_down_sync(0xffffffff, acc6, offset);
        acc7 += __shfl_down_sync(0xffffffff, acc7, offset);
    }
    
    // Write output (only lane 0) - vectorized for better memory bandwidth
    if (lane_id == 0) {
        half out_half[8];
        out_half[0] = __float2half(acc0);
        out_half[1] = __float2half(acc1);
        out_half[2] = __float2half(acc2);
        out_half[3] = __float2half(acc3);
        out_half[4] = __float2half(acc4);
        out_half[5] = __float2half(acc5);
        out_half[6] = __float2half(acc6);
        out_half[7] = __float2half(acc7);
        *reinterpret_cast<uint4*>(&Y[out_base]) = *reinterpret_cast<uint4*>(out_half);
    }
}

// ============================================================================
// GEMM Kernel (Prefill Phase, M>1) - Optimized with AWQ order
// ============================================================================

/**
 * Optimized AWQ GEMM kernel for prefill
 * Block handles [TILE_M, 8] outputs
 * 
 * Key optimizations:
 * - __ldg() for read-only cache
 * - Correct AWQ reverse order unpacking
 * - Precomputed -scale * zero for FMA
 */
__global__ __launch_bounds__(256, 2)
void awq_gemm_kernel(
    const half* __restrict__ X,           // [batch_size, in_features]
    const int32_t* __restrict__ qweight,  // [in_features, out_features/8]
    const int32_t* __restrict__ qzeros,   // [n_groups, out_features/8]
    const half* __restrict__ scales,      // [n_groups, out_features]
    half* __restrict__ Y,                 // [batch_size, out_features]
    int batch_size,
    int in_features,
    int out_features,
    int group_size
) {
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    const int num_warps = blockDim.x / 32;  // 8
    
    constexpr int TILE_M = 4;  // Each warp handles 4 batch elements
    
    const int batch_start = blockIdx.y * TILE_M;
    const int packed_out_idx = blockIdx.x * num_warps + warp_id;
    const int out_base = packed_out_idx * 8;
    
    if (out_base >= out_features) return;
    
    const int packed_out_dim = out_features / 8;
    const int n_groups = in_features / group_size;
    
    // AWQ reverse order for unpacking
    const int awq_order[8] = {0, 4, 1, 5, 2, 6, 3, 7};  // elem->pos mapping for vllm format
    
    // Accumulators for [TILE_M, 8] outputs
    float acc[TILE_M][8];
    #pragma unroll
    for (int m = 0; m < TILE_M; m++) {
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            acc[m][i] = 0.0f;
        }
    }
    
    // Process all groups
    for (int g = 0; g < n_groups; g++) {
        // Load zeros
        const int32_t qz_packed = __ldg(&qzeros[g * packed_out_dim + packed_out_idx]);
        
        // Load scales
        float s[8];
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            s[i] = __half2float(__ldg(&scales[g * out_features + out_base + i]));
        }
        
        // Extract zeros with AWQ order and precompute -scale * zero
        float neg_sz[8];
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            float z = (float)((qz_packed >> (awq_order[i] * 4)) & 0xF);
            neg_sz[i] = -s[i] * z;
        }
        
        // Process inputs in this group
        const int group_start = g * group_size;
        for (int k = lane_id; k < group_size; k += 32) {
            const int in_idx = group_start + k;
            
            // Load packed weights
            const int32_t w_packed = __ldg(&qweight[in_idx * packed_out_dim + packed_out_idx]);
            
            // Compute dequantized weights with AWQ order: scale * w + (-scale * zero)
            float dw[8];
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                float w = (float)((w_packed >> (awq_order[i] * 4)) & 0xF);
                dw[i] = s[i] * w + neg_sz[i];
            }
            
            // Accumulate for each batch element
            #pragma unroll
            for (int m = 0; m < TILE_M; m++) {
                int batch_idx = batch_start + m;
                if (batch_idx < batch_size) {
                    float x = __half2float(__ldg(&X[batch_idx * in_features + in_idx]));
                    #pragma unroll
                    for (int i = 0; i < 8; i++) {
                        acc[m][i] += x * dw[i];
                    }
                }
            }
        }
    }
    
    // Warp reduction
    #pragma unroll
    for (int m = 0; m < TILE_M; m++) {
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2) {
                acc[m][i] += __shfl_down_sync(0xffffffff, acc[m][i], offset);
            }
        }
    }
    
    // Write output
    if (lane_id == 0) {
        #pragma unroll
        for (int m = 0; m < TILE_M; m++) {
            int batch_idx = batch_start + m;
            if (batch_idx < batch_size) {
                #pragma unroll
                for (int i = 0; i < 8; i++) {
                    Y[batch_idx * out_features + out_base + i] = __float2half(acc[m][i]);
                }
            }
        }
    }
}

// ============================================================================
// Weight Dequantization Kernel for cuBLAS HGEMM
// ============================================================================

/*
 * Fast dequantization kernel: INT4 -> FP16
 * Each thread processes one packed INT32 (8 FP16 outputs)
 * Uses AWQ reverse order unpacking with vectorized memory access
 */
__global__ __launch_bounds__(256)
void awq_dequant_weight_kernel(
    const int32_t* __restrict__ qweight,  // [K, N/8] packed INT4
    const int32_t* __restrict__ qzeros,   // [K/G, N/8] packed INT4
    const half* __restrict__ scales,      // [K/G, N] FP16
    half* __restrict__ weight_fp16,       // [K, N] output
    int K,
    int N,
    int group_size
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int packed_N = N / 8;
    const int total_packed = K * packed_N;
    
    if (idx >= total_packed) return;
    
    const int k = idx / packed_N;
    const int n_packed = idx % packed_N;
    const int n_base = n_packed * 8;
    const int g = k / group_size;
    
    // AWQ reverse order for unpacking
    const int awq_order[8] = {0, 4, 1, 5, 2, 6, 3, 7};  // elem->pos mapping for vllm format
    
    // Load packed weight and zeros using __ldg for cache optimization
    int32_t w_packed = __ldg(&qweight[idx]);
    int32_t z_packed = __ldg(&qzeros[g * packed_N + n_packed]);
    
    // Load scales as uint4 for vectorized access
    uint4 scale_vec = *reinterpret_cast<const uint4*>(&scales[g * N + n_base]);
    half* scale_ptr = reinterpret_cast<half*>(&scale_vec);
    
    // Apply dequantization with AWQ order: (w - z) * scale
    half result[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int w = (w_packed >> (awq_order[i] * 4)) & 0xF;
        int z = (z_packed >> (awq_order[i] * 4)) & 0xF;
        result[i] = __hmul(__float2half((float)(w - z)), scale_ptr[i]);
    }
    
    // Store as uint4 for coalesced access
    *reinterpret_cast<uint4*>(&weight_fp16[k * N + n_base]) = *reinterpret_cast<uint4*>(result);
}

// Global cuBLAS handle for AWQ GEMM (lazy initialization)
static cublasHandle_t g_cublas_handle = nullptr;

// Static buffer for dequantized weights (avoid malloc/free per call)
// Pre-allocate maximum buffer size to avoid runtime allocation
static half* g_dequant_buffer = nullptr;
static size_t g_dequant_buffer_size = 0;
static bool g_initialized = false;

// Initialize global resources once
static void ensure_initialized() {
    if (g_initialized) return;
    
    // Create cuBLAS handle
    cublasCreate(&g_cublas_handle);
    cublasSetMathMode(g_cublas_handle, CUBLAS_TENSOR_OP_MATH);
    
    // Pre-allocate buffer for largest layer: 12288 * 4096 * sizeof(half) ≈ 100MB
    // For Qwen3-8B: max is W1/W2/W3 at [4096, 12288] or [12288, 4096]
    const size_t max_buffer_size = 12288 * 4096 * sizeof(half);
    cudaMalloc(&g_dequant_buffer, max_buffer_size);
    g_dequant_buffer_size = max_buffer_size;
    
    g_initialized = true;
}

// Flag to choose between vllm-style kernel and cuBLAS for prefill
// Set to true to use vllm-style MMA kernel (Tensor Core), false for cuBLAS fallback
static bool g_use_wmma_kernel = false;  // Back to cuBLAS for stability

void awq_gemm_tensorcore_cu(
    const half* input,
    const int32_t* qweight,
    const int32_t* qzeros,
    const half* scales,
    half* output,
    int M,
    int in_features,
    int out_features,
    int group_size,
    int split_k_iters,
    cudaStream_t stream
) {
    // Ensure global resources are initialized
    ensure_initialized();
    
    const int packed_out_dim = out_features / 8;
    
    // Version 6.0 WMMA Strategy:
    // - M=1 (decode): Use fast fused GEMV kernel (exploits INT4 bandwidth advantage)
    // - M>1 (prefill): Use WMMA-based W4A16 GEMM kernel (Tensor Cores)
    //
    // Key innovation: WMMA kernel dequantizes in shared memory and uses
    // Tensor Core for compute, achieving high throughput without extra memory.
    
    if (M == 1) {
        // GEMV for decode: use fast fused kernel
        awq_gemm_fast_cu(
            input, qweight, qzeros, scales, output,
            M, in_features, out_features, group_size, stream
        );
    } else if (true) {  // Enable vllm kernel for fast prefill
        // Prefill: use Tensor Core MMA kernel with fused dequant
        awq_gemm_vllm_cu(
            input, qweight, qzeros, scales, output,
            M, in_features, out_features, group_size, stream
        );
    } else {
        // Fallback: runtime dequant + cuBLAS HGEMM
        
        // Check if global dequant buffer is large enough
        const size_t required_size = (size_t)in_features * out_features * sizeof(half);
        if (g_dequant_buffer == nullptr || required_size > g_dequant_buffer_size) {
            if (g_dequant_buffer) cudaFree(g_dequant_buffer);
            cudaMalloc(&g_dequant_buffer, required_size);
            g_dequant_buffer_size = required_size;
        }
        
        // Step 1: Fast dequantization to temp buffer
        const int total_packed = in_features * packed_out_dim;
        const int dequant_threads = 256;
        const int dequant_blocks = (total_packed + dequant_threads - 1) / dequant_threads;
        
        awq_dequant_weight_kernel<<<dequant_blocks, dequant_threads, 0, stream>>>(
            qweight, qzeros, scales, g_dequant_buffer,
            in_features, out_features, group_size
        );
        
        // Step 2: cuBLAS HGEMM with Tensor Core
        const half alpha = __float2half(1.0f);
        const half beta = __float2half(0.0f);
        
        cublasSetStream(g_cublas_handle, stream);
        cublasSetMathMode(g_cublas_handle, CUBLAS_TENSOR_OP_MATH);
        
        cublasHgemm(
            g_cublas_handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            out_features,
            M,
            in_features,
            &alpha,
            g_dequant_buffer,
            out_features,
            input,
            in_features,
            &beta,
            output,
            out_features
        );
    }
}

// ============================================================================
// AWQ Weight Dequantization (standalone for pre-dequant)
// ============================================================================

void awq_dequant_weight_cu(
    const int32_t* qweight,      // [in_features, out_features/8]
    const int32_t* qzeros,       // [in_features/group_size, out_features/8]
    const half* scales,          // [in_features/group_size, out_features]
    half* dequant_weight,        // [in_features, out_features] output buffer
    int in_features,
    int out_features,
    int group_size,
    cudaStream_t stream
) {
    const int packed_out_dim = out_features / 8;
    const int total_packed = in_features * packed_out_dim;
    const int dequant_threads = 256;
    const int dequant_blocks = (total_packed + dequant_threads - 1) / dequant_threads;
    
    awq_dequant_weight_kernel<<<dequant_blocks, dequant_threads, 0, stream>>>(
        qweight, qzeros, scales, dequant_weight,
        in_features, out_features, group_size
    );
}

// ============================================================================
// cuBLAS-based AWQ GEMM for prefill (optimal for large M)
// ============================================================================

void awq_gemm_cublas_cu(
    const half* input,           // [M, in_features]
    const int32_t* qweight,      // [in_features, out_features/8]
    const int32_t* qzeros,       // [in_features/group_size, out_features/8]
    const half* scales,          // [in_features/group_size, out_features]
    half* dequant_weight,        // [in_features, out_features] temp buffer
    half* output,                // [M, out_features]
    int M,
    int in_features,
    int out_features,
    int group_size,
    cublasHandle_t cublas_handle,
    cudaStream_t stream
) {
    // Step 1: Dequantize weights to FP16
    const int packed_out_dim = out_features / 8;
    const int total_packed = in_features * packed_out_dim;
    const int dequant_threads = 256;
    const int dequant_blocks = (total_packed + dequant_threads - 1) / dequant_threads;
    
    awq_dequant_weight_kernel<<<dequant_blocks, dequant_threads, 0, stream>>>(
        qweight, qzeros, scales, dequant_weight,
        in_features, out_features, group_size
    );
    
    // Step 2: Use cuBLAS HGEMM with Tensor Core
    // C[M, out_features] = A[M, in_features] @ B[in_features, out_features]
    const half alpha = __float2half(1.0f);
    const half beta = __float2half(0.0f);
    
    // cuBLAS uses column-major, so we compute: C^T = B^T @ A^T
    // Which gives us row-major: C = A @ B
    cublasSetStream(cublas_handle, stream);
    cublasSetMathMode(cublas_handle, CUBLAS_TENSOR_OP_MATH);
    
    cublasHgemm(
        cublas_handle,
        CUBLAS_OP_T,          // B^T
        CUBLAS_OP_N,          // A (no transpose)
        out_features,         // m (output cols)
        M,                    // n (output rows)
        in_features,          // k
        &alpha,
        dequant_weight,       // B: [in_features, out_features]
        in_features,          // lda
        input,                // A: [M, in_features]
        in_features,          // ldb
        &beta,
        output,               // C: [M, out_features]
        out_features          // ldc
    );
}

// ============================================================================
// AWQ GEMM with pre-dequantized weights (fastest path for prefill)
// ============================================================================

void awq_gemm_with_dequant_cu(
    const half* input,           // [M, in_features]
    const half* dequant_weight,  // [in_features, out_features] pre-dequantized
    half* output,                // [M, out_features]
    int M,
    int in_features,
    int out_features,
    cudaStream_t stream
) {
    // Ensure cuBLAS handle is initialized
    ensure_initialized();
    
    cublasSetStream(g_cublas_handle, stream);
    
    const half alpha = __float2half(1.0f);
    const half beta = __float2half(0.0f);
    
    // Row-major C = A @ B using column-major trick
    // C[M, N] row-major = C^T[N, M] col-major
    // A[M, K] row-major = A^T[K, M] col-major
    // B[K, N] row-major = B^T[N, K] col-major
    // Compute: C^T = B^T @ A^T (in col-major)
    cublasHgemm(
        g_cublas_handle,
        CUBLAS_OP_N,          // B^T is [N, K] in col-major, no op needed
        CUBLAS_OP_N,          // A^T is [K, M] in col-major, no op needed
        out_features,         // m = N
        M,                    // n = M
        in_features,          // k = K
        &alpha,
        dequant_weight,       // B: [K, N] row-major
        out_features,         // ldb = N
        input,                // A: [M, K] row-major
        in_features,          // lda = K
        &beta,
        output,               // C: [M, N] row-major
        out_features          // ldc = N
    );
}

// ============================================================================
// Weight repacking: Kuiper AWQ format -> vllm AWQ format
// ============================================================================
/**
 * Kuiper AWQ packing: {0,4,1,5,2,6,3,7}
 *   bits[0:3]=elem0, bits[4:7]=elem4, bits[8:11]=elem1, bits[12:15]=elem5,
 *   bits[16:19]=elem2, bits[20:23]=elem6, bits[24:27]=elem3, bits[28:31]=elem7
 *
 * vllm AWQ packing: {0,2,4,6,1,3,5,7}
 *   bits[0:3]=elem0, bits[4:7]=elem2, bits[8:11]=elem4, bits[12:15]=elem6,
 *   bits[16:19]=elem1, bits[20:23]=elem3, bits[24:27]=elem5, bits[28:31]=elem7
 *
 * The repacking swaps element positions while keeping values intact.
 */
__global__ void awq_repack_weights_kernel(
    const int32_t* __restrict__ kuiper_weights,
    int32_t* __restrict__ vllm_weights,
    int total_packed
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_packed) return;
    
    uint32_t kw = kuiper_weights[idx];
    
    // Extract 8 INT4 values from Kuiper format
    int elem[8];
    // Kuiper: position -> element mapping
    elem[0] = (kw >> 0) & 0xF;   // pos 0 -> elem 0
    elem[4] = (kw >> 4) & 0xF;   // pos 1 -> elem 4
    elem[1] = (kw >> 8) & 0xF;   // pos 2 -> elem 1
    elem[5] = (kw >> 12) & 0xF;  // pos 3 -> elem 5
    elem[2] = (kw >> 16) & 0xF;  // pos 4 -> elem 2
    elem[6] = (kw >> 20) & 0xF;  // pos 5 -> elem 6
    elem[3] = (kw >> 24) & 0xF;  // pos 6 -> elem 3
    elem[7] = (kw >> 28) & 0xF;  // pos 7 -> elem 7
    
    // Repack into vllm format: {0,2,4,6,1,3,5,7}
    // vllm: position -> element
    // pos 0 -> elem 0, pos 1 -> elem 2, pos 2 -> elem 4, pos 3 -> elem 6
    // pos 4 -> elem 1, pos 5 -> elem 3, pos 6 -> elem 5, pos 7 -> elem 7
    uint32_t vw = 0;
    vw |= (elem[0] << 0);   // pos 0 <- elem 0
    vw |= (elem[2] << 4);   // pos 1 <- elem 2
    vw |= (elem[4] << 8);   // pos 2 <- elem 4
    vw |= (elem[6] << 12);  // pos 3 <- elem 6
    vw |= (elem[1] << 16);  // pos 4 <- elem 1
    vw |= (elem[3] << 20);  // pos 5 <- elem 3
    vw |= (elem[5] << 24);  // pos 6 <- elem 5
    vw |= (elem[7] << 28);  // pos 7 <- elem 7
    
    vllm_weights[idx] = vw;
}

void awq_repack_weights_cu(
    const int32_t* kuiper_weights,
    int32_t* vllm_weights,
    int K,
    int N,
    cudaStream_t stream
) {
    int packed_N = N / 8;
    int total_packed = K * packed_N;
    int threads = 256;
    int blocks = (total_packed + threads - 1) / threads;
    awq_repack_weights_kernel<<<blocks, threads, 0, stream>>>(
        kuiper_weights, vllm_weights, total_packed
    );
}

}  // namespace kernel

