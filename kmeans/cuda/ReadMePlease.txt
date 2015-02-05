Just a elementary implement of CUDA. 
On my desktop, it can at least accelerate the speed for 10 times.

1.You need to set up the system for CUDA.(Drivers,CUDAtoolkit etc.)

2.You can use the following command to compile and link:
nvcc --compiler-options '-fPIC' -I /usr/include/python2.7 -c kmeans_c_extension.cpp kmeans_chunk_center_cuda.cu

nvcc -shared -L /usr/local/cuda/lib64 -lcudart -lcuda -o kmeans_c_extension_cuda.so kmeans_c_extension.o kmeans_chunk_center_cuda.o

3.For this moment, you need to set the chunksize to be the same as the DATA_SIZE in the cuda implement, I'll fix it later.

4.I will do the optimization and debug in the following days.
