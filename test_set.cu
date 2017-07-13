#include <thrust/set_operations.h>
#include <thrust/host_vector.h>

#include <stdio.h>

int main()
{
	int A1[7] = {13, 1, 5, 3, 1, 1, 0};
	int A2[7] = {13, 8, 5, 3, 2, 1, 1};
	
	thrust::host_vector<int> result(7);
	thrust::host_vector<int>::iterator result_end;

	result_end = thrust::set_intersection(A1, A1 + 7, A2, A2 + 7, result.begin());
	
	//result.erase(result_end, result.end());

	for (int i = 0; i < result.size(); i++)
		printf("%d\n", result[i]);
}
