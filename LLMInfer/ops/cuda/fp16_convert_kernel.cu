#include "fp16_convert_kernel.cuh"
#include <cuda_fp16.h>

namespace kernel {

/**
 * Simple CUDA kernel to convert FP16 to FP32
 * Each thread handles one element
 */
__global__ void fp16_to_fp32_kernel(const half* __restrict__ input, 
                                     float* __restrict__ output, 
                                     size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = __half2float(input[idx]);
    }
}

/**
 * Vectorized FP16 to FP32 conversion kernel
 * Each thread handles 4 elements using half2 operations
 */
__global__ void fp16_to_fp32_vectorized_kernel(const half2* __restrict__ input,
                                                float4* __restrict__ output,
                                                size_t num_vec4) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_vec4) {
        // Read 4 FP16 values (2 half2)
        half2 h0 = input[idx * 2];
        half2 h1 = input[idx * 2 + 1];
        
        // Convert to float
        float2 f0 = __half22float2(h0);
        float2 f1 = __half22float2(h1);
        
        // Write as float4
        output[idx] = make_float4(f0.x, f0.y, f1.x, f1.y);
    }
}

void fp16_to_fp32_kernel_cu(const half* input, float* output, size_t size, cudaStream_t stream) {
    if (size == 0) return;
    
    // Check if we can use vectorized version (size must be multiple of 4)
    if (size % 4 == 0 && 
        reinterpret_cast<uintptr_t>(input) % 8 == 0 &&   // half2 aligned
        reinterpret_cast<uintptr_t>(output) % 16 == 0) { // float4 aligned
        const int threads = 256;
        size_t num_vec4 = size / 4;
        const int blocks = (num_vec4 + threads - 1) / threads;
        
        fp16_to_fp32_vectorized_kernel<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const half2*>(input),
            reinterpret_cast<float4*>(output),
            num_vec4
        );
    } else {
        // Fallback to element-wise conversion
        const int threads = 256;
        const int blocks = (size + threads - 1) / threads;
        
        fp16_to_fp32_kernel<<<blocks, threads, 0, stream>>>(input, output, size);
    }
}

void fp16_cpu_to_fp32_gpu(const uint16_t* input_cpu, float* output_gpu, size_t size, cudaStream_t stream) {
    if (size == 0) return;
    
    // For very large tensors, process in chunks to avoid allocating too much memory at once
    const size_t MAX_CHUNK_ELEMENTS = 64 * 1024 * 1024;  // 64M elements = 128MB for FP16, 256MB for FP32
    size_t remaining = size;
    size_t offset = 0;
    
    while (remaining > 0) {
        size_t chunk_size = std::min(remaining, MAX_CHUNK_ELEMENTS);
        size_t byte_size = chunk_size * sizeof(uint16_t);
        
        // Allocate page-locked (pinned) host memory for staging
        uint16_t* pinned_buffer = nullptr;
        cudaError_t err = cudaMallocHost(&pinned_buffer, byte_size);
        bool use_pinned = (err == cudaSuccess);
        
        if (!use_pinned) {
            // Fallback: try with regular malloc
            pinned_buffer = (uint16_t*)malloc(byte_size);
            if (!pinned_buffer) {
                printf("Malloc failed for staging buffer, chunk size: %zu\n", chunk_size);
                return;
            }
        }
        
        // Copy from mmap'd memory to staging buffer
        memcpy(pinned_buffer, input_cpu + offset, byte_size);
        
        // Allocate temporary GPU buffer for FP16 data
        half* temp_fp16_gpu = nullptr;
        err = cudaMalloc(&temp_fp16_gpu, byte_size);
        if (err != cudaSuccess) {
            printf("CUDA malloc failed: %s\n", cudaGetErrorString(err));
            if (use_pinned) cudaFreeHost(pinned_buffer);
            else free(pinned_buffer);
            return;
        }
        
        // Copy FP16 data from staging memory to GPU
        err = cudaMemcpy(temp_fp16_gpu, pinned_buffer, byte_size, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            printf("CUDA memcpy H2D failed: %s\n", cudaGetErrorString(err));
            cudaFree(temp_fp16_gpu);
            if (use_pinned) cudaFreeHost(pinned_buffer);
            else free(pinned_buffer);
            return;
        }
        
        // Convert FP16 to FP32 on GPU for this chunk
        fp16_to_fp32_kernel_cu(temp_fp16_gpu, output_gpu + offset, chunk_size, stream);
        
        // Wait for conversion to complete before freeing temp buffer
        if (stream) {
            cudaStreamSynchronize(stream);
        } else {
            cudaDeviceSynchronize();
        }
        
        // Free buffers
        cudaFree(temp_fp16_gpu);
        if (use_pinned) cudaFreeHost(pinned_buffer);
        else free(pinned_buffer);
        
        offset += chunk_size;
        remaining -= chunk_size;
    }
}

}  // namespace kernel
