#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <string.h>

#include <math.h>
#include <iostream>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#define cublasErrCheck(stat)                         \
    {                                                \
        cublasErrCheck_((stat), __FILE__, __LINE__); \
    }

void cublasErrCheck_(cublasStatus_t stat, const char *file, int line)
{
    if (stat != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "cuBLAS Error: %d %s %d\n", stat, file, line);
    }
}

__global__ void print_mat_dev(float *A, int m, int n)
{
    for (int i = 0; i < m; i++)
    {
        printf("Row %3d:", i);
        for (int j = 0; j < n; j++)
        {
            printf("%6.2f ", A[i * n + j]);
        }
        printf("\n");
    }
}

int main(int argc, char *argv[])
{

    if (argc < 4)
    {
        printf("Usage: %s <M> <N> <K> [check]\n", argv[0]);
        return 1;
    }

    int M = atoi(argv[1]);
    int N = atoi(argv[2]);
    int K = atoi(argv[3]);

    printf("M = %d, N = %d, K = %d\n", M, N, K);

    double *A = (double *)malloc(M * K * sizeof(double));
    double *B = (double *)malloc(K * N * sizeof(double));

#pragma omp parallel for
    for (int i = 0; i < M * K; i++)
    {
        A[i] = (int)rand() % 100;
    }
#pragma omp parallel for
    for (int i = 0; i < K * N; i++)
    {
        B[i] = (int)rand() % 100;
    }

    double *d_C;
    cudaMalloc(&d_C, M * N * sizeof(double));
    cudaMemset(d_C, 0, M * N * sizeof(double));

    double *d_A;
    cudaMalloc(&d_A, M * K * sizeof(double));
    cudaMemcpy(d_A, A, M * K * sizeof(double), cudaMemcpyHostToDevice);

    double *d_B;
    cudaMalloc(&d_B, K * N * sizeof(double));
    cudaMemcpy(d_B, B, K * N * sizeof(double), cudaMemcpyHostToDevice);

    cublasHandle_t handle;
    cublasCreate(&handle);

    double alpha = 1.0;
    double beta = 0.0;

    struct timeval start, end;

    cudaEvent_t d_start, d_stop;

    cudaEventCreate(&d_start);
    cudaEventCreate(&d_stop);

    cudaDeviceSynchronize();
    gettimeofday(&start, NULL);

    float gemm_time;
    cudaEventRecord(d_start, 0);
    cublasErrCheck(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                               N, M, K, &alpha,
                               d_B, N,
                               d_A, K,
                               &beta,
                               d_C, N));
    cudaEventRecord(d_stop, 0);
    cudaEventSynchronize(d_stop);
    cudaEventElapsedTime(&gemm_time, d_start, d_stop);

    cudaDeviceSynchronize();
    gettimeofday(&end, NULL);

    cudaError_t err = cudaGetLastError();

    if (err == cudaSuccess)
    {
        double gflops = 2.0 * M * N * K * 1e-9 / ((end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) * 1e-6);
        printf("Size: %d,%d,%d, Time: %f ms, GFLOPS: %f, gemm time: %f\n", M, N, K,
               (end.tv_sec - start.tv_sec) * 1e3 + (end.tv_usec - start.tv_usec) * 1e-3,
               2.0 * M * N * K * 1e-9 / ((end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) * 1e-6),
               gemm_time);
        printf("[hemeng_log]%d,%d,%d,%f,%f,%f\n", M, N, K, (end.tv_sec - start.tv_sec) * 1e3 + (end.tv_usec - start.tv_usec) * 1e-3, 2.0 * M * N * K * 1e-9 / ((end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) * 1e-6), gemm_time);
    }

    return 0;
}