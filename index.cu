#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
// #include <iostream>
// using namespace std;
__global__ void block_index(){
    // การคำนวณ Global Thread ID ใน รูปแบบ 3d
    // Grid -> Block -> Thread จาก x,y,z เป็น id 
    int block_id = 
    blockIdx.x +
    blockIdx.y * gridDim.x +
    blockIdx.z * gridDim.x * gridDim.y;
    // คำนวณ offset เริ่มต้นของ block นี้ มีกี่ thread
    int block_offset = block_id * (blockDim.x * blockDim.y * blockDim.z);
    // คำนวณ เลขที่ thread ใน block แบบ เชิงเส้น
    int thread_offset = 
    threadIdx.y +
    threadIdx.y * blockDim.x +
    threadIdx.z * blockDim.x * blockDim.y;
    int id = block_offset + thread_offset; // อยากรู้ id หนึ่งเนี่ยที่มีไม่ซ้ำกันทั้งระบบเท่าไหร่
    printf("%04d | Block(%d %d %d) = %3d| Thread(%d %d %d) = %3d\n",id,blockIdx.x,blockIdx.y,blockIdx.z,block_id,threadIdx.x,threadIdx.y,threadIdx.z,thread_offset);
}

int main(int argc,char **argv){
    const  int b_x =2 , b_y = 3, b_z =4; // 24 block
    const int t_x = 4 , t_y = 4 , t_z = 4; //64 threads

    int block_per_grid = b_x*b_y*b_z;
    int threads_per_block = t_x * t_y *t_z;

    printf("%d blocks/grid\n",block_per_grid);
    printf("%d threads/block\n",threads_per_block);
    printf("%d total threads\n",block_per_grid*threads_per_block);
    // kernel ข้อมูล 3 มิติ นำค่าเข้าตัวแปร แล้วรันใน ฟักชัน block index
    dim3 blocksPerGrid(b_x,b_y,b_z); // 24
    dim3 threadsPerBlock(t_x,t_y,t_z); // 64

    block_index <<< blocksPerGrid, threadsPerBlock>>>();
    // รันให้ GPU ทำงานเสร็จ  แล้วค่อยจบโปรแกรม ไม่ต้อง return 0; ตาม C
    cudaDeviceSynchronize();

}