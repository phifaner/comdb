#include <stdio.h>
#include <cassert>

#define ALLOC_SIZE 128

__global__ void
test_malloc(int **controller)
{
	__shared__ int *ptr;
	int bx = blockIdx.x;
	
	if (threadIdx.x == 0)
	{
		ptr = (int*)malloc(ALLOC_SIZE * sizeof(int));
		controller[bx] = ptr;
		
		printf("allocate GPU memory at %d\n", ptr);		
	}

	__syncthreads();

	for (int idx = threadIdx.x; idx < ALLOC_SIZE; idx += blockDim.x)
	{
		ptr[idx] = threadIdx.x;
	}
}

__global__ void
test_free(int **controller)
{
	int bx = blockIdx.x;
	if (threadIdx.x == 0)
	{
		free(controller[bx]);
		printf("free controller of %d\n", bx);
	}
}

int main()
{
	int block_num = 64;
	int block_size = 32;
	int **g_controller;
	
	cudaMalloc(&g_controller, sizeof(int*)*block_num);
	test_malloc<<<block_num, block_size>>>(g_controller);

	int *h_controller[block_num];
	cudaMemcpy(h_controller, g_controller, sizeof(int*)*block_num, cudaMemcpyDeviceToHost);
	
	for (int i = 0; i != block_size; i++)
	{
		printf("allocated pointer %p. \n", h_controller[i]);
		int buffer[ALLOC_SIZE];
		
		cudaMemcpy(buffer, h_controller[i], sizeof(int)*ALLOC_SIZE, cudaMemcpyDeviceToHost);
		
		for (int index = 0; index != block_size; index++) ;
			//assert(buffer[index] == index);

	}
	
	test_free<<<block_num, block_size>>>(g_controller);
}
