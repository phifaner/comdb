#include "project.h"
#include "BasicDefinitions.h"
#include "Random.h"

extern long long totalAllocatedMemory;

// Hash function list
PHashFunctionT hashFunctionList;

// Initialize the hash functions. c is approximation ratio
void init_hash_function(int mbase, int dim, double max_coord, double c, double w) {

	// bEnd = (c^Floor(log(c)(t*d))) * w^2
	double bEnd = (int) ((log(dim) + log(max_coord)) / log(c));
	bEnd = pow(c, bEnd) * w * w;

	FAILIF(NULL==(hashFunctionList=(PHashFunctionT)MALLOC(mbase * sizeof(HashFunction))));
	for (int i = 0; i < mbase; i++) {
		FAILIF(NULL==(hashFunctionList[i].a =(double*)MALLOC(dim * sizeof(double))));
		// Each dimensional value of vector <a> is chosen from Gaussian Distribution.
		for (int j = 0; j < dim; j++) {
			hashFunctionList[i].a[j] = genGaussianRandom();
		}

		// <b> is randomly chosen from Uniform Distribution U(0, (c^Floor(log(c)(t*d))) * w^2).
		hashFunctionList[i].b = genUniformRandom(0, bEnd);

		//std::cout << "---------------" << hashFunctionList[i].a[2] << ", " << hashFunctionList[i].b << std::endl;
	}

}

int* proj_points(int mbase, int dim, size_t length, int wsize, thrust::device_vector<int> &win_id_vec, 
	thrust::device_vector<double> &win_lon_vec, thrust::device_vector<double> &win_lat_vec)
{
	thrust::device_vector<int> device_hash_table(mbase*length/wsize);

	// wsize -- sliding-window size --8
	// length -- trajectory length
	//	std::cout << "---------------" << length << std::endl;

	//thrust::device_ptr<int> table_ptr = thrust::device_malloc<int>(mbase*length*wsize);
	int *p_device_hash_table = thrust::raw_pointer_cast(device_hash_table.data());
	
	std::cout << "----length: " << length << " mbase: " << mbase << " dimiension: " << dim << " window size: " << wsize  << "hash talbe size: " << mbase*length/wsize << std::endl;

	thrust::device_vector<double> hash_function_a_vec(mbase*dim);	
	thrust::device_vector<double> hash_function_b_vec(mbase);

	for (int i = 0; i < mbase; i++)
	{
		hash_function_b_vec[i] = hashFunctionList[i].b;

		for (int j = 0; j < dim; j++) hash_function_a_vec[i*dim+j] = hashFunctionList[i].a[j];
	}

	project proj(thrust::raw_pointer_cast(hash_function_a_vec.data()), thrust::raw_pointer_cast(hash_function_b_vec.data()),
			mbase, dim, p_device_hash_table, length, 
			thrust::raw_pointer_cast(win_id_vec.data()), thrust::raw_pointer_cast(win_lon_vec.data()), 
			thrust::raw_pointer_cast(win_lat_vec.data()));

	thrust::for_each
	(
		thrust::counting_iterator<unsigned int>(0),
		thrust::counting_iterator<unsigned int>(length/wsize),
		proj
	);

	int *host_hash_table;
  	host_hash_table = (int*) malloc(sizeof(int)*mbase*length/wsize);
	cudaMemcpy(host_hash_table, p_device_hash_table, mbase*length/wsize, cudaMemcpyDeviceToHost);

	//std::cout << "--------------------------" << host_hash_table[100] << std::endl;

	return host_hash_table;
}

/*void to_inv_list(GPUGenie_Config& config, inv_table& table, int *hash_data)
{

	
}*/
