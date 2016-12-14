#include "project.h"
#include "BasicDefinitions.h"
#include "Random.h"

extern long long totalAllocatedMemory;

// Hash function list
PHashFunctionT hashFunctionList;

// Initialize the hash functions. c is approximation ratio
void init_hash_function_old(int mbase, int dim, double max_coord, double c, double w) {
	// initialize the random seed
	srand((unsigned) time(NULL));

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

void init_hash_function(int mbase, int dim, double radius)
{
	srand((unsigned) time(NULL));

	FAILIF(NULL==(hashFunctionList=(PHashFunctionT)MALLOC(mbase * sizeof(HashFunction))));
	
	for (int i = 0; i < mbase; i++)
	{
		FAILIF(NULL==(hashFunctionList[i].a =(double*)MALLOC(dim * sizeof(double))));
		
		for (int j = 0; j < dim; j++)
		{
			hashFunctionList[i].a[j] = genGaussianRandom();
			
			
			//if (i == 10 && j == 5) std::cout << "hash function --------" << hashFunctionList[i].a[j] << "\t";
		}
		
		hashFunctionList[i].b = genUniformRandom(0, radius);
	}

}

void proj_points(int mbase, int dim, size_t length, int wsize, int *host_hash_data, thrust::device_vector<int> &win_id_vec, 
	thrust::device_vector<double> &win_lon_vec, thrust::device_vector<double> &win_lat_vec, 
		double min_lon, double max_lon, double min_lat, double max_lat, double w)
{
	std::cout << "double min_lon = " << min_lon << ", max_lon = " << max_lon << ", min_lat = " << min_lat << ", max_lat = " << max_lat << std::endl;

	thrust::device_vector<int> device_hash_table(mbase*length/wsize);

	// wsize -- sliding-window size --8
	// length -- trajectory length
	int *p_device_hash_table = thrust::raw_pointer_cast(device_hash_table.data());
	//int *p_device_id_table = thrust::raw_pointer_cast(device_id_table.data());
	
	//std::cout << "----length: " << length << " mbase: " << mbase << " dimiension: " << dim << " window size: " << wsize  << "hash talbe size: " << mbase*length/wsize << std::endl;

	thrust::device_vector<double> hash_function_a_vec(mbase*dim);	
	thrust::device_vector<double> hash_function_b_vec(mbase);

	for (int i = 0; i < mbase; i++)
	{
		hash_function_b_vec[i] = hashFunctionList[i].b;

		for (int j = 0; j < dim; j++) 
		{
			hash_function_a_vec[i*dim+j] = hashFunctionList[i].a[j];
		
			//if (i == 10 && j == 5)	std::cout << "copy hash function ---" << i*dim+j  << ": " << hash_function_a_vec[i*dim+j] << std::endl;
		}
	}

	project proj(thrust::raw_pointer_cast(hash_function_a_vec.data()), thrust::raw_pointer_cast(hash_function_b_vec.data()),
			mbase, dim, p_device_hash_table, length, 
			thrust::raw_pointer_cast(win_id_vec.data()), thrust::raw_pointer_cast(win_lon_vec.data()), 
			thrust::raw_pointer_cast(win_lat_vec.data()), min_lon, max_lon, min_lat, max_lat, w);

	thrust::for_each
	(
		thrust::counting_iterator<unsigned int>(0),
		thrust::counting_iterator<unsigned int>(length/wsize),
		proj
	);

	// copy data into host
  	//host_hash_data = (int*) malloc(sizeof(int)*mbase*length/wsize);
	thrust::copy(device_hash_table.begin(), device_hash_table.end(), host_hash_data);
	
	//thrust::copy(device_hash_table.begin(), device_hash_table.end(), std::ostream_iterator<int>(std::cout, ","));
	
	//host_id_data = (int*) malloc(sizeof(int)*mbase*length/wsize);
	//thrust::copy(device_id_table.begin(), device_id_table.end(), host_id_data);

	std::cout << "mbase:" << mbase << " length: " << length << " wsize: " << wsize << std::endl;
	//for (int i = 0; i < mbase*length/wsize; i++)
	//	if (i == 199+ 200*125) std::cout <<  host_hash_data[i] << "\t";
	
	//return host_hash_table;
}

void to_inv_list(GPUGenie::GPUGenie_Config& config, GPUGenie::inv_table *table, int *hash_data, int data_size, int dim){
	GPUGenie::inv_list list;
	int i, j;
	
	//std::cout << "----data size: ------" << data_size << " dimension: " << dim << std::endl;
	for (i = 0; i < dim; i++)
	{
		std::vector<int> col;
		col.reserve(data_size);
		
		for (j = 0; j < data_size; j++)
		{
			//std::cout << hash_data[j*dim+i] << "\t";
			
			col.push_back(hash_data[j*dim+i]);
		}
		
		//std::cout << "push back-----------------------: " << col.size() << std::endl;
	
		list.invert(col);
		//std::cout << "invert ----------------: " << col.size() << std::endl;
		table->append(list);
	}	
		//std::cout << "build index-----------------------" << std::endl;

	table->build(config.posting_list_max_length, config.use_load_balance);
	
	if (config.save_to_gpu)	table->cpy_data_to_gpu();
	table->is_stored_in_gpu = config.save_to_gpu;
}

