#include <iostream>
#include <cuda_runtime.h>

__global__ void process_spectrogram_matrix_2d(float* d_matrix, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int index = y * width + x;

        d_matrix[index] = d_matrix[index] * 2.0f;
    }
}

int main() {
    const int WIDTH = 16;
    const int HEIGHT = 16;
    const int N = WIDTH * HEIGHT;
    size_t bytes = N * sizeof(float);

    std::cout << "[STEP 1] Creating CPU 2D matrix (" << WIDTH << "x" << HEIGHT << ")...\n";
    float* h_matrix = new float[N];
    float* h_out = new float[N];

    for (int y = 0; y < HEIGHT; ++y) {
        for (int x = 0; x < WIDTH; ++x) {
            int idx = y * WIDTH + x;
            h_matrix[idx] = static_cast<float>(idx);
            h_out[idx] = 0.0f;
        }
    }

    std::cout << "[STEP 2] Allocating GPU memory (" << bytes << " bytes)...\n";
    float* d_matrix = nullptr;
    cudaError_t err = cudaMalloc((void**)&d_matrix, bytes);
    if (err != cudaSuccess) {
        std::cerr << "[CUDA ERROR] cudaMalloc failed: " << cudaGetErrorString(err) << "\n";
        delete[] h_matrix; delete[] h_out;
        return 1;
    }

    std::cout << "[STEP 3] Copying input data from CPU to GPU...\n";
    err = cudaMemcpy(d_matrix, h_matrix, bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "[CUDA ERROR] cudaMemcpy H2D failed: " << cudaGetErrorString(err) << "\n";
        cudaFree(d_matrix); delete[] h_matrix; delete[] h_out;
        return 1;
    }

    dim3 threadsPerBlock(8, 8);
    dim3 numBlocks((WIDTH + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (HEIGHT + threadsPerBlock.y - 1) / threadsPerBlock.y);

    std::cout << "[STEP 4] Launching 2D Kernel with "
              << numBlocks.x << "x" << numBlocks.y << " blocks ("
              << threadsPerBlock.x << "x" << threadsPerBlock.y << " threads)...\n";

    process_spectrogram_matrix_2d<<<numBlocks, threadsPerBlock>>>(d_matrix, WIDTH, HEIGHT);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "[CUDA ERROR] Kernel launch failed: " << cudaGetErrorString(err) << "\n";
        cudaFree(d_matrix); delete[] h_matrix; delete[] h_out;
        return 1;
    }

    std::cout << "[STEP 5] Copying processed results back to CPU...\n";
    err = cudaMemcpy(h_out, d_matrix, bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "[CUDA ERROR] cudaMemcpy D2H failed: " << cudaGetErrorString(err) << "\n";
        cudaFree(d_matrix); delete[] h_matrix; delete[] h_out;
        return 1;
    }

    bool pass = true;
    for (int i = 0; i < N; ++i) {
        if (h_out[i] != 2.0f * h_matrix[i]) {
            pass = false;
            break;
        }
    }

    std::cout << "\n--- TOP-LEFT CORNER (3x3) ---" << std::endl;
    for (int y = 0; y < 3; ++y) {
        for (int x = 0; x < 3; ++x) {
            int idx = y * WIDTH + x;
            std::cout << h_out[idx] << "\t";
        }
        std::cout << std::endl;
    }

    if (pass) {
        std::cout << "\nSTATUS: PASS (All 2D matrix elements correctly doubled!)\n";
    } else {
        std::cout << "\nSTATUS: FAIL\n";
    }

    cudaFree(d_matrix);
    delete[] h_matrix;
    delete[] h_out;

    return 0;
}