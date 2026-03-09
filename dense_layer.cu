#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

// เช็ค error ของ cuda
#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// Activation Function (ReLU)
__device__ float relu(float x) {
    return fmaxf(0.0f, x);
}

// Kernel: Dense Layer (Matrix Mul + Bias + ReLU)
// Input: (batch_size, input_dim)
// Weight: (input_dim, output_dim)
// Bias: (output_dim,)
// Output: (batch_size, output_dim)
__global__ void denseLayerForward(float *input, float *weight, float *bias, float *output, 
                                  int batch_size, int input_dim, int output_dim) {
    
    // คำนวณ Index ของ Output Matrix (row, col)
    int row = blockIdx.y * blockDim.y + threadIdx.y; // batch index
    int col = blockIdx.x * blockDim.x + threadIdx.x; // output neuron index

    if (row < batch_size && col < output_dim) {
        float sum = 0.0f;
        
        // Dot Product: Input[row] • Weight[:, col]
        for (int k = 0; k < input_dim; k++) {
            sum += input[row * input_dim + k] * weight[k * output_dim + col];
        }

        // Add Bias
        sum += bias[col];

        // Activation (ReLU)
        output[row * output_dim + col] = relu(sum);
    }
}

// Kernel: คำนวณ Loss (Mean Squared Error)
__global__ void calculateLoss(float *output, float *target, float *loss, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float diff = output[i] - target[i];
        // Atomic add เพื่อรวมค่า loss จากทุก thread
        atomicAdd(loss, diff * diff);
    }
}

int main() {

    int batch_size = 4;
    int input_dim = 2;
    int output_dim = 1;

    float h_input[] = {0, 0,  0, 1,  1, 0,  1, 1};
    float h_target[] = {0,  1,  1,  0};
    
    // Weight และ Bias (สุ่มค่าเริ่มต้น)
    float h_weight[] = {0.5f, 0.5f}; 
    float h_bias[] = {0.1f};         // 1

    float *d_input, *d_weight, *d_bias, *d_output, *d_target, *d_loss;
    float h_loss = 0.0f;

    int size_input = batch_size * input_dim * sizeof(float);
    int size_weight = input_dim * output_dim * sizeof(float);
    int size_bias = output_dim * sizeof(float);
    int size_output = batch_size * output_dim * sizeof(float);

    // Allocate Memory
    cudaCheckError(cudaMalloc(&d_input, size_input));
    cudaCheckError(cudaMalloc(&d_weight, size_weight));
    cudaCheckError(cudaMalloc(&d_bias, size_bias));
    cudaCheckError(cudaMalloc(&d_output, size_output));
    cudaCheckError(cudaMalloc(&d_target, batch_size * output_dim * sizeof(float)));
    cudaCheckError(cudaMalloc(&d_loss, sizeof(float)));

    // Copy Data
    cudaCheckError(cudaMemcpy(d_input, h_input, size_input, cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(d_weight, h_weight, size_weight, cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(d_bias, h_bias, size_bias, cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(d_target, h_target, batch_size * output_dim * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(d_loss, &h_loss, sizeof(float), cudaMemcpyHostToDevice));

    // Configure Grid/Block ปรับ Grid และ Block ที่จะจ้องพื้นที่ 
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((output_dim + 15) / 16, (batch_size + 15) / 16);

    // Run Forward Pass
    denseLayerForward<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_weight, d_bias, d_output, 
                                                          batch_size, input_dim, output_dim);
    
    // Calculate Loss
    cudaCheckError(cudaMemset(d_loss, 0, sizeof(float))); // Reset loss
    int total_elements = batch_size * output_dim;
    calculateLoss<<<(total_elements + 255)/256, 256>>>(d_output, d_target, d_loss, total_elements);

    // Copy Loss back
    cudaCheckError(cudaMemcpy(&h_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost));
    h_loss /= total_elements; // Mean

    printf("Loss start: %f\n", h_loss);

    // Free Memory
    cudaFree(d_input); cudaFree(d_weight); cudaFree(d_bias); 
    cudaFree(d_output); cudaFree(d_target); cudaFree(d_loss);

    return 0;
}