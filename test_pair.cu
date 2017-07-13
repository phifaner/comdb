#include <thrust/pair.h>
#include <thrust/transform.h>
#include <thrust/host_vector.h>
#include <thrust/copy.h>
#include <thrust/sort.h>
#include <thrust/device_vector.h>

#define N 10

struct make_pair_functor
{
	template<typename T>
	__host__ __device__
	thrust::pair<T, T> operator() (const T &x, const T &y)
	{
		return thrust::make_pair(x, y);
	}
};

struct pair_to_vector_first
{
	template<typename T>
	__host__ __device__
	int operator() (const T &x)
	{
		return x.first;
	}
};

struct pair_to_vector_second
{
	template<typename T>
	__host__ __device__
	int operator() (const T &x)
	{
		return x.second;
	}
};

struct accumulate_diff
{
	int *keys;
	int *values;
	int *counts;

	accumulate_diff(int *k, int *v, int *c) : keys(k), values(v), counts(c) {}

	template<typename T>
	__device__
	void operator() (const T &i)	
	{
		__shared__ volatile int _sd[N];
		_sd[i] = counts[i];
	
		//__threadfence_system();

		if (i == 0)  counts[keys[i]] = 1; //__threadfence_system();}
		if (i > 0) 
		{
			//while (keys[i] != keys[i-1] && _sd[i] == 0);
			
			if (keys[i] != keys[i-1])
			{
				 //_sd[i] = 1;
				 atomicAdd(&counts[keys[i]], 1); 
				//__threadfence_system();
				//_sd[i] = 0;
			}
			else				 
			{
				if (values[i] != values[i-1]) 
				{
					//printf("777777777777777 %d\n", keys[i]);
					//_sd[i] = 1;
					atomicAdd(&counts[keys[i]], 1);
					//__threadfence_system();
					//_sd[i] = 0;
				}
			}
		}
		
		//__threadfence_system();

		//counts[keys[i]] = _sd[keys[i]];
	}
};


int main()
{
	int A[N] = {1, 3, 3, 3, 3, 2, 1, 2, 2, 1};
	int B[N] = {9, 8, 7, 5, 6, 7, 8, 7, 6, 9};

	int C[N];
	int D[N];
	thrust::device_vector<int> counts(N, 0);

	typedef thrust::pair<int, int> P;
	thrust::host_vector<P> h_pairs(N);

	thrust::transform(A, A+N, B, h_pairs.begin(), make_pair_functor());

	thrust::sort(h_pairs.begin(), h_pairs.end());

	thrust::transform(h_pairs.begin(), h_pairs.end(), C, pair_to_vector_first());
	thrust::transform(h_pairs.begin(), h_pairs.end(), D, pair_to_vector_second());

	thrust::device_vector<int> c_vec(C, C+N);
	thrust::device_vector<int> d_vec(D, D+N);
	//thrust::reduce_by_key(C, C+7, thrust::constant_iterator<int>(1), C, )
	accumulate_diff acc(thrust::raw_pointer_cast(c_vec.data()), thrust::raw_pointer_cast(d_vec.data()), thrust::raw_pointer_cast(counts.data()));

	thrust::for_each(thrust::counting_iterator<unsigned int>(0), thrust::counting_iterator<unsigned int>(N), acc);

	for (int i = 0; i < N; i++)
	{
		std::cout << h_pairs[i].first << ": " << h_pairs[i].second << "\n";
		std::cout << "=========" << C[i] << "\n";
		std::cout << "++++++++++++++" << counts[i] << "\n";
	}
}
