#include <getopt.h>
#include <memory>
#include<iostream>
#include <string>

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <rocblas.h>
#include <hiprand_kernel.h>
#include <hip/hip_fp16.h> // For __half and __float2half
#include <cstring> // For memcpy


struct Args {
    int m = 1024;
    int n = 1024;
    int k = 1024;
    int warmup = 20;
    int iter = 50;
};

void process_args(int argc, char **argv, Args *args) {
    const char *const short_opts = "m:n:k:w:i:";
    const option long_opts[] = {
        {"warmup", required_argument, nullptr, 'w'},
        {"iter", required_argument, nullptr, 'i'},
    };

    int opt = 0;
    while ((opt = getopt_long(argc, argv, short_opts, long_opts, nullptr)) != -1) {
        switch (opt) {
        case 'm':
            args->m = std::stoi(optarg);
            break;
        case 'n':
            args->n = std::stoi(optarg);
            break;
        case 'k':
            args->k = std::stoi(optarg);
            break;
        case 'w':
            args->warmup = std::stoi(optarg);
            break;
        case 'i':
            args->iter = std::stoi(optarg);
            break;
        }
    }
}

// Kernel to initialize a matrix with random values between -1 and 1
__global__ void init_matrix(rocblas_half* matrix, size_t size, unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        // Initialize random state for the thread
        hiprandState state;
        hiprand_init(seed, idx, 0, &state);

        // Generate random value in [0, 1], scale to [-1, 1]
        float random_value = hiprand_uniform(&state) * 2.0f - 1.0f;

        // Convert float to __half
        __half temp_half = __float2half(random_value);

        // Assign the __half value to rocblas_half
        matrix[idx].data = *reinterpret_cast<uint16_t*>(&temp_half);
    }
}


class GemmOp {
    public:
        hipError_t err;
        // Constructor to initialize the dimensions of the matrices and the alpha and beta values
        GemmOp(int m, int n, int k, int warmup, int iter) : m_(m), n_(n), k_(k), alpha_(1.0f), beta_(0.0f), warmup_(warmup), iter_(iter) {}
        // Function to allocate memory on device and initialize matrices A and B
        void Setup() {
            rocblas_create_handle(&handle_);
            // allocate memory on device
            err = hipMalloc(&dA_, m_ * k_ * sizeof(rocblas_half));
            if (err != hipSuccess) {
                fprintf(stderr, "hipMalloc failed: %s\n", hipGetErrorString(err));
            }
            
            err = hipMalloc(&dB_, k_ * n_ * sizeof(rocblas_half));
            if (err != hipSuccess) {
                fprintf(stderr, "hipMalloc failed: %s\n", hipGetErrorString(err));
            }
            
            err = hipMalloc(&dC_, m_ * n_ * sizeof(rocblas_half));
            if (err != hipSuccess) {
                fprintf(stderr, "hipMalloc failed: %s\n", hipGetErrorString(err));
            }

            // Initialize matrices A and B with random values between -1 and 1
            unsigned long long seed = 1234; // Seed for reproducibility
            int threadsPerBlock = 256;
            int blocksA = (m_ * k_ + threadsPerBlock - 1) / threadsPerBlock;
            int blocksB = (k_ * n_ + threadsPerBlock - 1) / threadsPerBlock;
            // Call kernel to initialize matrices A and B
            init_matrix<<<blocksA, threadsPerBlock>>>(dA_, m_ * k_, seed);
            init_matrix<<<blocksB, threadsPerBlock>>>(dB_, k_ * n_, seed);
            (void)hipDeviceSynchronize(); // Ensure initialization completes before proceeding

            // verify the initialization
            rocblas_half *hA = new rocblas_half[m_ * k_];
            rocblas_half *hB = new rocblas_half[k_ * n_];

            hipMemcpy(hA, dA_, m_ * k_ * sizeof(rocblas_half), hipMemcpyDeviceToHost);
            hipMemcpy(hB, dB_, k_ * n_ * sizeof(rocblas_half), hipMemcpyDeviceToHost);

            for (int i = 0; i < m_ * k_ ; i++) {
                __half temp_half = *reinterpret_cast<__half*>(&hA[i].data);
                float temp_float = __half2float(temp_half);
                if (temp_float > 1.0f || temp_float < -1.0f) {
                    printf("Matrix A initialization failed\n");
                    break;
                }
            }

            delete[] hA;
            delete[] hB;

        }
        // Function to execute the GEMM operation
        void Execute()
        {
            rocblas_status status = rocblas_gemm_ex(
                handle_,
                rocblas_operation_transpose,
                rocblas_operation_none,
                m_, n_, k_,
                &alpha_,
                dA_, rocblas_datatype_f16_r, k_,
                dB_, rocblas_datatype_f16_r, k_,
                &beta_,
                dC_, rocblas_datatype_f16_r, m_,
                dC_, rocblas_datatype_f16_r, m_,
                rocblas_datatype_f32_r,
                rocblas_gemm_algo_standard, 0, 0);

            if (status != rocblas_status_success) {
                fprintf(stderr, "rocblas_gemm_ex failed with status %d\n", status);
            }
        }
        // Destructor to deallocate memory on device
        ~GemmOp() {
            (void)hipFree(dA_);
            (void)hipFree(dB_);
            (void)hipFree(dC_);
            (void)rocblas_destroy_handle(handle_);
        }

    private:

        int m_, n_, k_, warmup_, iter_;
        float alpha_, beta_;
        rocblas_half *dA_, *dB_, *dC_;
        rocblas_handle handle_;
};

int main(int argc, char **argv) {
    Args args;
    // Process command line arguments
    process_args(argc, argv, &args);
    // Create GemmOp object
    GemmOp op(args.m, args.n, args.k, args.warmup, args.iter);
    // allocates memory on device and initializes matrices
    op.Setup();
    
    // warmup
    for (int i = 0; i < args.warmup; i++) {
        op.Execute();
    }
    // Wait for the warmup to complete
    (void)hipDeviceSynchronize(); // Ensure warmup completes before proceeding

    float time;
    hipEvent_t start, stop;
    (void)hipEventCreate(&start);
    (void)hipEventCreate(&stop);

    (void)hipEventRecord(start, 0);
    for (int i = 0; i < args.iter; i++) {
        op.Execute();
    }
    (void)hipEventRecord(stop, 0);
    (void)hipEventSynchronize(stop);
    (void)hipEventElapsedTime(&time, start, stop);

    // deallocate resouces
    (void)hipEventDestroy(start);
    (void)hipEventDestroy(stop);

    float time_us = ((time * 1e3) / (float) args.iter);
    printf("%d\t%d\t%d\t%d\t%f\t%f\n", args.m, args.n, args.k, 1, time_us,
           float(args.m) * float(args.n) * float(2 * args.k - 1) / 1e6 / time_us * 1);
    return 0;
}
