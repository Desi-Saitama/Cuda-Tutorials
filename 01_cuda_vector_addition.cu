#include<stdio.h>
#include<stdlib.h>
#include<cuda_runtime.h> //cuda runtime library

#define N 100000
//size of vector

__global__ void vector_add(float *a, float *b, float *c, int n){
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if(tid < n){
		c[tid] = a[tid] + b[tid];
	}
}

void initialize_vec(float *vec, int n){
	for(int i=0; i<n; i++){
		vec[i] = (float)rand()/RAND_MAX;
	}
}

int main(){
	printf("Run Started \n");
	
	float *h_a, *h_b, *h_c;
	float *d_a, *d_b, *d_c;
	size_t size = N*sizeof(float);
	
	//allocate host memory
	h_a = (float*)malloc(size);
	h_b = (float*)malloc(size);
	h_c = (float*)malloc(size);
	
	//initialise vector to a radnom value
	initialize_vec(h_a, N);
	initialize_vec(h_b, N);
	
	//allocate memory in device
	cudaMalloc(&d_a, size);
	cudaMalloc(&d_b, size);
	cudaMalloc(&d_c, size);
	
	//copy data from host to device
	cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
	
	//define grid and block
	int threads_per_block = 256;
	int blocks_per_grid = (N+threads_per_block-1)/threads_per_block;
	
	//calling the kernel
	vector_add<<<blocks_per_grid, threads_per_block>>>(d_a, d_b, d_c, N);
	cudaDeviceSynchronize();
	
	// Copy result back to host
  	cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);
	
	printf("Run Successful \n");
	printf("Result :: first 10 elements \n");
	for(int i=0; i<10; i++){
		printf("%f ", h_c[i]);
	}
	
	//free used memory;
	free(h_a);
	free(h_b);
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
	
	return 0;
}
