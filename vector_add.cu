#include <stdio.h>
#include <cuda_runtime.h>

// กำหนดขนาดของข้อมูล
#define N 10000

// Macro สำหรับตรวจสอบ Error (สำคัญมาก!)
#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

//  Kernel Function (ฟังก์ชันที่จะรันบน GPU)
// คำสั่ง __global__ บอกว่าฟังก์ชันนี้เรียกจาก CPU แต่รันบน GPU
__global__ void vectorAdd(float *a, float *b, float *c, int n) {
    // คำนวณ Index ของข้อมูลว่า Thread นี้ต้องดูแลตัวที่เท่าไหร่
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // ตรวจสอบขอบเขต(ป้องกันไม่ให้เกินขนาดอาร์เรย์)
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    // ตัวแปรบน Host (CPU)
    float *h_a, *h_b, *h_c; 
    // ตัวแปรบน Device (GPU)
    float *d_a, *d_b, *d_c; 

    int size = N * sizeof(float);

    // จอง Memory บน Host
    h_a = (float*)malloc(size);
    h_b = (float*)malloc(size);
    h_c = (float*)malloc(size);

    //  ใส่ค่าเริ่มต้น (เช่น 1.0 + 2.0)
    for (int i = 0; i < N; i++) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }

    // จอง Memory บน Device (GPU)
    cudaCheckError( cudaMalloc((void**)&d_a, size) );
    cudaCheckError( cudaMalloc((void**)&d_b, size) );
    cudaCheckError( cudaMalloc((void**)&d_c, size) );

    // Copy ข้อมูลจาก Host -> Device
    // cudaMemcpy(ปลายทาง, ต้นทาง, ขนาด, ทิศทาง)
    cudaCheckError( cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice) );
    cudaCheckError( cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice) );

    // กำหนดจำนวน Thread และ Block
    int threadsPerBlock = 256;
    // คำนวณจำนวน Block ที่ต้องการ (ปัดเศษขึ้น)
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    //  เรียกใช้ Kernel
    // Syntax: ชื่อKernel<<<จำนวนBlock, จำนวนThreadต่อBlock>>>(พารามิเตอร์)
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);

    // Copy ผลลัพธ์จาก Device -> Host
    cudaCheckError( cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost) );

    // ตรวจสอบผลลัพธ์ (แสดงแค่ 5 ตัวแรก)
    printf("first 5 element:\n");
    for (int i = 0; i < 5; i++) {
        printf("%f + %f = %f\n", h_a[i], h_b[i], h_c[i]);
    }

    // คืนหน่วยความจำ (Free Memory)
    cudaCheckError( cudaFree(d_a) );
    cudaCheckError( cudaFree(d_b) );
    cudaCheckError( cudaFree(d_c) );
    free(h_a);
    free(h_b);
    free(h_c);

    printf("Compelete!\n");

    return 0;
}