#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>

#define TILE_SIZE 16

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

__global__ void gemm_tiled_kernel(int M, int N, int K, const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    float sum = 0.0f;
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; t++) {
        // Cooperatively load tile into Shared Memory (SRAM)
        int a_col = t * TILE_SIZE + tx;
        int b_row = t * TILE_SIZE + ty;

        if (row < M && a_col < K) {
            As[ty][tx] = A[row * K + a_col];
        } else {
            As[ty][tx] = 0.0f;
        }

        if (b_row < K && col < N) {
            Bs[ty][tx] = B[b_row * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }

        __syncthreads();

        // Compute dot product on the local SRAM tile
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

int main(int argc, char** argv) {
    int M = (argc > 1) ? atoi(argv[1]) : 1024;
    int N = (argc > 2) ? atoi(argv[2]) : 1024;
    int K = (argc > 3) ? atoi(argv[3]) : 1024;
    int device_id = (argc > 4) ? atoi(argv[4]) : 4;
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

    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 blocksPerGrid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    // Warmup
    for (int w = 0; w < 3; w++) {
        gemm_tiled_kernel<<<blocksPerGrid, threadsPerBlock>>>(M, N, K, d_A, d_B, d_C);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    printf("{\"operator\":\"matmul\",\"subtype\":\"tiled\",\"params\":{\"M\":%d,\"N\":%d,\"K\":%d,\"TILE\":%d},\"latencies_ms\":[", M, N, K, TILE_SIZE);

    for (int it = 0; it < iters; it++) {
        CUDA_CHECK(cudaEventRecord(start));
        gemm_tiled_kernel<<<blocksPerGrid, threadsPerBlock>>>(M, N, K, d_A, d_B, d_C);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        printf("%s%.4f", (it > 0 ? "," : ""), ms);
    }

    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes_C, cudaMemcpyDeviceToHost));
    bool valid = true;
    float expected = 0.25f * K;
    for (int i = 0; i < 50; i++) {
        if (fabs(h_C[i] - expected) > 1e-2) {
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

    return 0;
}
