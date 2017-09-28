__global__ void add(int *a, int *b)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	a[index] += b[index];
}


int main()
{
	int *device_array_a = 0 ;
	int *device_array_b = 0;
	int h_array_a[5] = {1,2,3,4,5};
	int h_array_b[5] = {1,2,3,4,5};

	int num_bytes = 5 * sizeof(int);
	cudaMalloc((void**)&device_array_a, num_bytes);
	cudaMalloc((void**)&device_array_b, num_bytes);

	cudaMemcpy(device_array_a, h_array_a, num_bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(device_array_b, h_array_b, num_bytes, cudaMemcpyHostToDevice);

	add<<<5, 1>>>(device_array_a, device_array_b);

	cudaMemcpy(h_array_a, device_array_a, num_bytes, cudaMemcpyDeviceToHost);

	printf("%d, %d\n", h_array_a[1], h_array_a[2]);

	cudaFree(device_array_a);
	cudaFree(device_array_b);
}
