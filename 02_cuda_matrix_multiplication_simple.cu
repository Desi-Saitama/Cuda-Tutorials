#include<stdio.h>
#include<stdlib.h>
#include<cuda_runtime.h>

#define M 256
// Number of rows in matrix A and C
#define K 512
// Number of columns in A and rows in B
#define N 256
// Number of columns in B and C
#define BLOCK_SIZE 32

//kernel
__global__ void matmul_gpu(float *A, float *B, float *C, int m, int k, int n){
	int row = blockIdx.y*blockDim.y + threadIdx.y;
	int col = blockIdx.x*blockDim.x + threadIdx.x;
	
	if(row<m && col<n){
		float sum = 0.0f;
		for(int l=0; l<k; l++){
			sum += A[row*k+l] * B[l*n+col];
		}
		C[row*n+col] = sum;
	}
}

//intialise matrix
void init_matrix(float *matrix, int row, int col){
	for(int i=0; i<row*col; i++){
		matrix[i] = (float)rand()/RAND_MAX;
		}
}

int main(){
	printf("Execution Started\n");
	// Initialize all the variables
	float *h_A, *h_B, *h_C;
	float *d_A, *d_B, *d_C;
	int size_A = M*K*sizeof(float);
	int size_B = K*N*sizeof(float);
	int size_C = M*N*sizeof(float);
	
	//Allocate Memory in cpu
	h_A = (float*)malloc(size_A);
	h_B = (float*)malloc(size_B);
	h_C = (float*)malloc(size_C);
	
	// Initliase matrix
	init_matrix(h_A, M, K);
	init_matrix(h_B, K, N);
	
	// Allocate memory in device
	cudaMalloc(&d_A, size_A);
	cudaMalloc(&d_B, size_B);
	cudaMalloc(&d_C, size_C);
	
	//copy data from host to device
	cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);
	
	//define grid and block dimensions
	dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
	dim3 gridDim((N+BLOCK_SIZE-1)/BLOCK_SIZE, (M+BLOCK_SIZE-1)/BLOCK_SIZE);
	
	// calling the kernel
	matmul_gpu<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
	//syncgronize
	cudaDeviceSynchronize();
	
	cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);
	
	printf("Excution Finished\n");

	//for(int i=1; i<=M*N; i++){ if(i%N==0){
	//		printf("\n");
	//	}
	//	printf("%0.1f ", h_C[i]);
	//}
	free(h_A);
	free(h_B);
	free(h_C);
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	
  return 0;	
}
