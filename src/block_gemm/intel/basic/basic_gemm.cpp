#include <string>
#include <vector>
#include <cstdlib>
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>

#define BLOCKS_PER_GRID 1000
#define NUM_ITER 1000

#define ALPHA 1
#define BETA 0

using DATA_TYPE_AB = sycl::half;
using DATA_TYPE_C = float;

void init(DATA_TYPE_AB *A, DATA_TYPE_AB *B, DATA_TYPE_C *C, size_t size)
{
    for (size_t i = 0; i < size; i++)
    {
        for (size_t j = 0; j < size; j++)
        {
            A[i * size + j] = ((DATA_TYPE_AB)i * j) / size;
            B[i * size + j] = ((DATA_TYPE_AB)i * j + 1) / size;
            C[i * size + j] = 0;
        }
    }
}

void gemm_cpu(DATA_TYPE_AB *A, DATA_TYPE_AB *B, DATA_TYPE_C *C, size_t size)
{
    for (size_t i = 0; i < size; i++)
    {
        for (size_t j = 0; j < size; j++)
        {

            for (size_t k = 0; k < size; ++k)
            {

                C[i * size + j] = A[i * size + k] * B[k * size + j];
            }
        }
    }
}

void gemm_gpu(sycl::queue &queue, DATA_TYPE_AB *A, DATA_TYPE_AB *B, DATA_TYPE_C *C, size_t size)
{
    sycl::buffer<DATA_TYPE_AB, 2> A_buffer(A, sycl::range<2>(size, size));
    sycl::buffer<DATA_TYPE_AB, 2> B_buffer(B, sycl::range<2>(size, size));
    sycl::buffer<DATA_TYPE_C, 2> C_buffer(C, sycl::range<2>(size, size));

    sycl::range<2> size_1(1, BLOCKS_PER_GRID);
    sycl::range<2> size_2(size, size);

    auto e = queue.submit([&](sycl::handler &cgh)
                          {
        auto A_acc = A_buffer.get_access<sycl::access::mode::read>(cgh);
        auto B_acc = B_buffer.get_access<sycl::access::mode::read>(cgh);
        auto C_acc = C_buffer.get_access<sycl::access::mode::read_write>(cgh);

        cgh.parallel_for(
            sycl::range<3>(BLOCKS_PER_GRID, size, size), [=](sycl::item<3> item) 
        {
            const auto i = item[1];
            const auto j = item[2];

            
            for(size_t idx_iter = 0; idx_iter < NUM_ITER; idx_iter++) {
                for(size_t k = 0; k < size; k++) {
                    
                    C_acc[{i,j}] = A_acc[{i, k}] * B_acc[{k, j}];
                }
            }

        }); });

    queue.wait();

    auto kernel_duration = (e.get_profiling_info<sycl::info::event_profiling::command_end>() - e.get_profiling_info<sycl::info::event_profiling::command_start>());
    std::cout << "Kernel Execution Time : " << kernel_duration / 1e+9 << " seconds\n";
    std::cout << "Kernel Throughput      : " << (2.0 * size * size * size * BLOCKS_PER_GRID * NUM_ITER) / kernel_duration << " GFlops\n";
}

using namespace sycl;

void mm_kernel(queue &q, std::vector<float> &matrix_a, std::vector<float> &matrix_b, std::vector<float> &matrix_c, size_t N, size_t M)
{
    std::cout << "Configuration         : MATRIX_SIZE= " << N << "x" << N << "\n";

    buffer a(matrix_a);
    buffer b(matrix_b);
    buffer c(matrix_c);

    auto e = q.submit([&](handler &h)
                      {
        
        accessor A(a, h, read_only);
        accessor B(b, h, read_only);
        accessor C(c, h, write_only);

        
        h.parallel_for(range<2>{N,N}, [=](item<2> item){
            const int i = item.get_id(0);
            const int j = item.get_id(1);
            for (int k = 0; k < N; k++) {
                C[i*N+j] += A[i*N+k] * B[k*N+j];
            }
        }); });
    host_accessor hc(c, read_only);

    auto kernel_duration = (e.get_profiling_info<info::event_profiling::command_end>() - e.get_profiling_info<info::event_profiling::command_start>());
    std::cout << "[example] Kernel Execution Time : " << kernel_duration / 1e+9 << " seconds\n";
    std::cout << "[example] Kernel Throughput      : " << (2.0 * N * N * N) / kernel_duration << " GFlops\n";
}

void example()
{
    size_t N = 1024;
    size_t M = 1024;

    std::vector<float> matrix_a(N * N);
    std::vector<float> matrix_b(N * N);
    std::vector<float> matrix_c(N * N);
    std::vector<float> matrix_d(N * N);

    float v1 = 2.f;
    float v2 = 3.f;
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
        {
            matrix_a[i * N + j] = v1++;
            matrix_b[i * N + j] = v2++;
            matrix_c[i * N + j] = 0.f;
            matrix_d[i * N + j] = 0.f;
        }

    queue q(property::queue::enable_profiling{});
    std::cout << "Offload Device        : " << q.get_device().get_info<info::device::name>() << "\n";
    std::cout << "max_work_group_size   : " << q.get_device().get_info<info::device::max_work_group_size>() << "\n";

    auto start = std::chrono::high_resolution_clock::now().time_since_epoch().count();

    mm_kernel(q, matrix_a, matrix_b, matrix_c, N, M);
}

bool verify_result(DATA_TYPE_C *cpu_result, DATA_TYPE_C *gpu_result, size_t size)
{
    constexpr float ERROR_THRESHOLD = 0.05;
    for (size_t i = 0; i < size * size; i++)
    {
        float diff = std::abs(cpu_result[i] - gpu_result[i]) / cpu_result[i];
        if (diff > ERROR_THRESHOLD)
            return false;
    }
    return true;
}

int main(int argc, char **argv)
{
    int device_id = 0;
    dpct::device_info prop;

    dpct::select_device(device_id);
    dpct::get_device(device_id).get_device_info(prop);
    std::cout << "GPU " << prop.get_device_id() << " Model: " << prop.get_name()
              << std::endl;

    size_t size = argc > 1 ? std::atoi(argv[1]) : 64;

    std::vector<DATA_TYPE_AB> A(size * size);
    std::vector<DATA_TYPE_AB> B(size * size);
    std::vector<DATA_TYPE_C> C_gpu(size * size);
    std::vector<DATA_TYPE_C> C_cpu(size * size);

    init(A.data(), B.data(), C_gpu.data(), size);
    std::copy(C_gpu.begin(), C_gpu.end(), C_cpu.begin());

    sycl::queue queue(sycl::default_selector_v, sycl::property::queue::enable_profiling{});

    gemm_gpu(queue, A.data(), B.data(), C_gpu.data(), size);

    gemm_cpu(A.data(), B.data(), C_cpu.data(), size);

    example();

    std::cout << "Matrix Size: " << size << "x" << size << std::endl;

    bool is_correct = verify_result(C_cpu.data(), C_gpu.data(), size);
    std::cout << "Verification: " << (is_correct ? "PASSED" : "FAILED") << std::endl;

    return 0;
}