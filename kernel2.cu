#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <math.h>
#include <iostream>

#define N 10000000
#define BLOCK_SIZE_1D 1024
#define BLOCK_SIZE_3D_X 16
#define BLOCK_SIZE_3D_Y 8
#define BLOCK_SIZE_3D_Z 8

//จะเปรียบเทียบ ประสิทธิภาพ ระหว่าง CPU และ GPU ในการบวก เวกเตอร์ 10 ล้านตัวพร้อมกัน

// cpu vector addition
void vector_add_cpu(float *a, float *b, float *c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

// CUDA kernel for vector addition 
__global__ void vector_add_gpu(float *a, float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; // block ใน memmory

    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

// CUDA kernel for vector addition 3d
__global__ void vector_add_gpu_3d(float *a, float *b, float *c, int nx , int ny , int nz) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; // block ใน memmory
    int j = blockIdx.y * blockDim.y + threadIdx.y; // block ใน memmory
    int k = blockIdx.z * blockDim.z + threadIdx.z; // block ใน memmory
    // 3 multiples , 3 stores
    if (i < nx && j < ny && k < nz) {
        int idx = i + j * nx + k * nx * ny;
        if(idx < nx * ny * nz){
            c[idx] = a[idx] + b[idx];
        }
    }
}

// กำหนดค่า ตัวแปรเวกเตอร์แบบสุ่ม
void init_vector(float *vec, int n) {
    for (int i = 0; i < n; i++) {
        vec[i] = (float)rand() / RAND_MAX;
    }
}

// Function to measure execution time จาก lib time.h 
// version unix/linux
// double get_time() {
//     struct timespec ts;
//     clock_gettime(CLOCK_MONOTONIC, &ts);
//     return ts.tv_sec + ts.tv_nsec * 1e-9;
// }

double get_time(){

    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;

    cudaEventElapsedTime(&ms , start , stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return ms * 1e-3;
}

int main() {

    float *h_a, *h_b, *h_c_cpu, *h_c_gpu , *h_c_gpu_3d;
    float *d_a, *d_b, *d_c , *d_c_3d;
    size_t size = N * sizeof(float);

    // Allocate host memory หรือ การจองพื้นที่เพิ่มสำหรับตัวแปร
    h_a = (float*)malloc(size);
    h_b = (float*)malloc(size);
    h_c_cpu = (float*)malloc(size);
    h_c_gpu = (float*)malloc(size);
    h_c_gpu_3d = (float*)malloc(size);

    // Initialize vectors
    srand(time(NULL));
    init_vector(h_a, N);
    init_vector(h_b, N);

    // Allocate device memory หรือ จองพิ้นที่บน GPU
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);
    cudaMalloc(&d_c_3d, size);

    // Copy data to device นำข้อมูลเข้าพื้นที่ใน memory
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    int num_blocks = (N + BLOCK_SIZE_1D - 1) / BLOCK_SIZE_1D;
    // N = 1024, BLOCK_SIZE = 256, num_blocks = 4
    // (N + BLOCK_SIZE - 1) / BLOCK_SIZE = ( (1025 + 256 - 1) / 256 ) = 1280 / 256 = 4 rounded 
    // Define grid and block dimensions 3D
    int nx = 100 , ny = 100 , nz = 1000;
    dim3 block_size_3d(BLOCK_SIZE_3D_X, BLOCK_SIZE_3D_Y , BLOCK_SIZE_3D_Z);
    dim3 num_blocks_3d(
        (nx + block_size_3d.x - 1) / block_size_3d.x,
        (ny + block_size_3d.y - 1) / block_size_3d.y,
        (nz + block_size_3d.z - 1) / block_size_3d.z
    );

    // Warm-up runs
    printf("Performing warm-up runs...\n");
    for (int i = 0; i < 3; i++) {
        vector_add_cpu(h_a, h_b, h_c_cpu, N);
        vector_add_gpu<<<num_blocks, BLOCK_SIZE_1D>>>(d_a, d_b, d_c, N);
        vector_add_gpu_3d<<<num_blocks_3d, block_size_3d>>>(d_a, d_b, d_c, nx, ny, nz);
        cudaDeviceSynchronize();
    }

    // Benchmark CPU implementation
    printf("Benchmarking CPU implementation...\n");
    double cpu_total_time = 0.0;
    for (int i = 0; i < 5; i++) {
        double start_time = get_time();
        vector_add_cpu(h_a, h_b, h_c_cpu, N);
        double end_time = get_time();
        cpu_total_time += end_time - start_time;
    }
    double cpu_avg_time = cpu_total_time / 5.0;

    // Benchmark GPU implementation
    printf("Benchmarking GPU implementation...\n");
    double gpu_total_time = 0.0;
    for (int i = 0; i < 100; i++) {
        cudaMemset(d_c, 0 ,size);
        double start_time = get_time();
        vector_add_gpu<<<num_blocks, BLOCK_SIZE_1D>>>(d_a, d_b, d_c, N);
        cudaDeviceSynchronize();
        double end_time = get_time();
        gpu_total_time += end_time - start_time;
    }
    double gpu_avg_time = gpu_total_time / 100.0;

    // Verify results (optional)
    cudaMemcpy(h_c_gpu, d_c, size, cudaMemcpyDeviceToHost);
    bool correct = true;
    for (int i = 0; i < N; i++) {
        if (fabs(h_c_cpu[i] - h_c_gpu[i]) > 1e-5) {
            correct = false;
            std::cout << i << " cpu: " << h_c_cpu[i] << " != " << h_c_gpu[i] << std::endl;
            break;
        }
    }
    printf("Results are %s\n", correct ? "correct" : "incorrect");
    // Benchmark GPU 3D
    double gpu_3d_total_time = 0.0;
    for(int i = 0 ; i < 100 ; i++){
        cudaMemset(d_c_3d , 0 ,size);
        double start_time = get_time();
        vector_add_gpu_3d<<<num_blocks_3d,block_size_3d>>>(d_a,d_b,d_c_3d, nx , ny,nz);
        cudaDeviceSynchronize();
        double end_time = get_time();
        gpu_3d_total_time += end_time - start_time;
    }
    double gpu_3d_avg_time = gpu_3d_total_time / 100.0;
    cudaMemcpy(h_c_gpu_3d, d_c_3d , size , cudaMemcpyDeviceToHost);
    bool correct_3d = true;
    for (int i = 0 ; i < N ; i++){
        if(fabs(h_c_cpu[i] - h_c_gpu_3d[i]) > 1e-4){
            correct_3d = false;
            std::cout << i << " cpu: " << h_c_cpu[i] << " != " << h_c_gpu_3d[i] << std::endl;
            break;
        }
    }
    printf("3D result are %s\n",correct_3d ? "correct" : "incorrect");
    // Print results
    printf("CPU average time: %f milliseconds\n", cpu_avg_time * 1000);
    printf("GPU 1D average time: %f milliseconds\n", gpu_avg_time * 1000);
    printf("GPU 3D average time: %f milliseconds\n", gpu_3d_avg_time * 1000);
    printf("Speedup (CPU vs GPU 1D): %fx\n", cpu_avg_time / gpu_avg_time);
    printf("Speedup (CPU vs GPU 3D): %fx\n", cpu_avg_time / gpu_3d_avg_time);
    printf("Speedup (GPU 1D vs GPU 3D): %fx\n", gpu_avg_time / gpu_3d_avg_time);
    // Free memory นำตัวแปรออกจาก memory
    free(h_a);
    free(h_b);
    free(h_c_cpu);
    free(h_c_gpu);
    free(h_c_gpu_3d);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFree(d_c_3d);
    
    return 0;
}