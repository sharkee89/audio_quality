#include <iostream>
#include <cuda_runtime.h>

__global__ void double_array_kernel(float* d_arr, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = idx; i < n; i += stride) {
        d_arr[i] = d_arr[i] * 2.0f;
    }
}

int main() {
    const int N = 10000;
    size_t bytes = N * sizeof(float);

    std::cout << "[STEP 1] Creating CPU array a[i] = i..." << std::endl;
    float* h_a = new float[N];
    float* h_out = new float[N];

    for (int i = 0; i < N; ++i) {
        h_a[i] = static_cast<float>(i);
        h_out[i] = 0.0f;
    }

    std::cout << "[STEP 2] Allocating GPU memory (" << bytes << " bytes)..." << std::endl;
    float* d_arr = nullptr;
    cudaError_t err = cudaMalloc((void**)&d_arr, bytes);
    if (err != cudaSuccess) {
        std::cerr << "[CUDA ERROR] cudaMalloc failed: " << cudaGetErrorString(err) << std::endl;
        delete[] h_a;
        delete[] h_out;
        return 1;
    }

    std::cout << "[STEP 3] Copying input data from CPU to GPU..." << std::endl;
    err = cudaMemcpy(d_arr, h_a, bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "[CUDA ERROR] cudaMemcpy H2D failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_arr);
        delete[] h_a;
        delete[] h_out;
        return 1;
    }

    int threads_per_block = 256;
    int blocks = 4;

    std::cout << "[STEP 4] Launching kernel with " << blocks << " blocks of "
              << threads_per_block << " threads..." << std::endl;

    double_array_kernel<<<blocks, threads_per_block>>>(d_arr, N);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "[CUDA ERROR] Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_arr);
        delete[] h_a;
        delete[] h_out;
        return 1;
    }

    std::cout << "[STEP 5] Copying processed results back to CPU..." << std::endl;
    err = cudaMemcpy(h_out, d_arr, bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "[CUDA ERROR] cudaMemcpy D2H failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_arr);
        delete[] h_a;
        delete[] h_out;
        return 1;
    }

    bool pass = true;
    for (int i = 0; i < N; ++i) {
        if (h_out[i] != 2.0f * i) {
            pass = false;
            break;
        }
    }

    std::cout << "First 5 values from GPU: ";
    for (int i = 0; i < 5; ++i) {
        std::cout << h_out[i] << " ";
    }
    std::cout << std::endl;

    if (pass) {
        std::cout << "STATUS: PASS (All values are correct 2 * i)" << std::endl;
    } else {
        std::cout << "STATUS: FAIL (Data don't fit)" << std::endl;
    }

    cudaFree(d_arr);
    delete[] h_a;
    delete[] h_out;

    return 0;
}
