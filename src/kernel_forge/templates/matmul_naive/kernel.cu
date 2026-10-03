#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

__global__ void gemm_naive_kernel(int M, int N, int K, const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

int main(int argc, char** argv) {
    int M = (argc > 1) ? atoi(argv[1]) : 1024;
    int N = (argc > 2) ? atoi(argv[2]) : 1024;
    int K = (argc > 3) ? atoi(argv[3]) : 1024;
    int device_id = (argc > 4) ? atoi(argv[4]) : 0;
    int iters = (argc > 5) ? atoi(argv[5]) : 10;

    CUDA_CHECK(cudaSetDevice(device_id));

    size_t bytes_A = M * K * sizeof(float);
    size_t bytes_B = K * N * sizeof(float);
    size_t bytes_C = M * N * sizeof(float);

    float *h_A = (float*)malloc(bytes_A);
    float *h_B = (float*)malloc(bytes_B);
    float *h_C = (float*)malloc(bytes_C);

    for (int i = 0; i < M * K; i++) h_A[i] = 0.5f;
    for (int i = 0; i < K * N; i++) h_B[i] = 0.5f;

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc((void**)&d_A, bytes_A));
    CUDA_CHECK(cudaMalloc((void**)&d_B, bytes_B));
    CUDA_CHECK(cudaMalloc((void**)&d_C, bytes_C));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes_B, cudaMemcpyHostToDevice));

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + 15) / 16, (M + 15) / 16);

    // Warmup
    for (int w = 0; w < 3; w++) {
        gemm_naive_kernel<<<blocksPerGrid, threadsPerBlock>>>(M, N, K, d_A, d_B, d_C);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    printf("{\"operator\":\"matmul\",\"subtype\":\"naive\",\"params\":{\"M\":%d,\"N\":%d,\"K\":%d},\"latencies_ms\":[", M, N, K);

    for (int it = 0; it < iters; it++) {
        CUDA_CHECK(cudaEventRecord(start));
        gemm_naive_kernel<<<blocksPerGrid, threadsPerBlock>>>(M, N, K, d_A, d_B, d_C);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        printf("%s%.4f", (it > 0 ? "," : ""), ms);
    }

    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes_C, cudaMemcpyDeviceToHost));
    bool valid = true;
    float expected = 0.25f * K;
    for (int i = 0; i < M * N; i++) {
        if (!isfinite(h_C[i]) || fabs(h_C[i] - expected) > 1e-2) {
            valid = false;
            break;
        }
    }

    printf("],\"valid\":%s}\n", valid ? "true" : "false");

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C);

    return valid ? 0 : 1;
}
