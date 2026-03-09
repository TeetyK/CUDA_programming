Compile C++
```bash
g++ --fiile.cpp -o --output.exe
.\output
```
Compile NVCC CUDA
```bash
nvcc -arch=sm_86 -o file_cu file.cu
nvcc -arch=sm_86 -o cuda_test cuda_test.cu -allow-unsupported-compiler
.\cuda_test
```