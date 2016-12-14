#ifndef PROJECT_H_
#define PROJECT_H_

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>

#include "GPUGenie.h"

// A simple point in d-dimensional space. A point is defined by a
// vector of coordinates.
typedef struct _PointT {
	int index; // the index of this point in the dataset list of points
	double *coordinates;
//	double sqrLength; // the square of the length of the vector
} PointT, *PPointT;

// Hash function structure
typedef struct _HashFunction {
	// Vector a in LSH function
	double *a;

	// Real number b in LSH function
	double b;
} HashFunction, *PHashFunctionT;

// Hash function list
//PHashFunctionT hashFunctionList;

void init_hash_function(int mbase, int dim, /*double max_coord, double c, double w*/ double radius);

void proj_points(int mbase, int dim, size_t length, int wsize, int* host_hash_data, thrust::device_vector<int> &win_id_vec, 
	thrust::device_vector<double> &win_lon_vec, thrust::device_vector<double> &win_lat_vec, 
		double min_lon, double max_lon, double min_lat, double max_lat, double w);

void to_inv_list(GPUGenie::GPUGenie_Config& config, GPUGenie::inv_table *table, int *hash_data, int data_size, int dim);

__inline__ __host__ __device__
double min_max_norm(double x, double max, double min)
{
	return (x - min) / (max - min); 
}

struct project
{
	size_t length;
	double *hash_a_data, *hash_b_data;
	int func_num, dimension;
	double min_lon, max_lon, min_lat, max_lat;
	int *hash_table;
	//int *hash_id_data;	// to store the trajectory id for the knn search
	int *win_id_data;
	double *win_lon_data, *win_lat_data;
	double w;

	project(double *_a_data, double *_b_data, int _func_num, int _dimension, int *_hash_table,  
		size_t _len, int *id_data, double *lon_data, double *lat_data, double _min_lon, 
		double _max_lon, double _min_lat, double _max_lat, double _w) 
		: hash_a_data(_a_data), hash_b_data(_b_data), func_num(_func_num), dimension(_dimension), 
			hash_table(_hash_table), length(_len), win_id_data(id_data), win_lon_data(lon_data), 
				win_lat_data(lat_data), min_lon(_min_lon), max_lon(_max_lon), min_lat(_min_lat),
					max_lat(_max_lat), w(_w) {}

	//template <typename T>
	__host__ __device__
	void operator()(const int t)
	{
		int wsize = dimension/2; // or 4
		// the first id is same as the last id
		if (win_id_data[wsize*t] != win_id_data[wsize*t+wsize])	return;

		thrust::for_each(thrust::device, thrust::counting_iterator<unsigned int>(0), thrust::counting_iterator<unsigned int>(func_num), [=]__device__(const int &i) 
		{
			double value = 0;
			int interval_id = -1;	
		
			// Compute the inner product value of <a> * dataPoint.
			for (int m = 0; m < wsize; m++) 
			{
				// min-max normalization
				double lon = win_lon_data[wsize*t+m];
				double lat = win_lat_data[wsize*t+m];
				double norm_lon = min_max_norm(lon, max_lon, min_lon);
				double norm_lat = min_max_norm(lat, max_lat, min_lat);

				//if (hash_a_data[i*dimension+2*m] < 0 || hash_a_data[i*dimension+2*m+1] < 0) printf("---norm lon %lf, norm lat %lf\n", hash_a_data[i*dimension+2*m], hash_a_data[i*dimension+2*m+1]);
				//printf("--hash a data: %lf, hash a data 2: %lf, norm_lon: %lf, norm_lat: %lf\t", hash_a_data[i*dimension+m], hash_a_data[i*dimension+2*m], norm_lon, norm_lat);
				value += (hash_a_data[i*dimension+2*m] * norm_lon);	// add data of lontitude;
				value += (hash_a_data[i*dimension+2*m+1] * norm_lat);
			}

			value += hash_b_data[i];
			value /= w;

			interval_id = (int)value;
	
			// put into the hash tables, a 1-dimension array, store tid
			hash_table[t*func_num+i] = interval_id;
			//hash_id_data[t*func_num+i] = win_id_data[dimension*t];
			
			if (interval_id > 10000 || interval_id < -10000)
			{
				for (int m = 0; m < dimension; m++)	printf("hash a data: %lf\t, hash a data 2: %lf \n", hash_a_data[i*dimension+2*m], hash_a_data[i*dimension+2*m+1]);
			}
				//printf("============hash value: %d, hash a: %lf,  hash a 2: %lf, hash b: %lf \n", interval_id, win_lon_data[t*dimension], win_lat_data[t*dimension], hash_b_data[i]);

		});
	}
	
};

#endif
