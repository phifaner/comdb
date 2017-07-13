#include <thrust/device_vector.h>
#include <thrust/for_each.h>
#include <thrust/execution_policy.h>

struct print
{
	int *B;
	int len;

	print(int *b, int _len) : B(b), len(_len) {}

	__host__ __device__
	void operator() (int x)
	{
		thrust::for_each(thrust::device, B, B+len, [=](const int k) {
			if (k < len) printf("%d\n", B[x]);
		});
	}
};

void test()
{
	int A[6] = {0, 1, 2, 3, 4, 5};
	thrust::device_vector<int> vec(A, A+6);
	int B[6] = {0, 0, 0, 0, 0, 0};
	
	//thrust::for_each(vec.begin(), vec.end(), [=]__device__(int &i) {
		//thrust::for_each(vec.begin(), vec.end(), [=] __host__ __device__(const int& k) { 
	//		printf("%d\n", A[i]);
		//});
	//);
	thrust::for_each(vec.begin(), vec.end(), print(A, 6));
}

int main()
{
	test();
}
