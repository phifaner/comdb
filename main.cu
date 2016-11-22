#include <dirent.h>

#include <thrust/count.h>
#include <thrust/sort.h>
#include <thrust/extrema.h>

#include "comdb.h"
#include "swindow.h"
#include "project.h"
#include "GPUGenie.h"

using namespace std;

long long int totalAllocatedMemory = 0;

int main()
{
	cudaSetDevice(1);
	
	comdb db;
	
	// record time
	cudaEvent_t start, end;
	float time;
	
	cudaEventCreate(&start);
	cudaEventCreate(&end);

	cudaEventRecord(start, 0);

	// read all files in the folder
	DIR *dir;
	struct dirent *ent;
	char path[50];

	if ((dir = opendir("./data")) != NULL)
	{
		while ((ent = readdir(dir)) != NULL)
		{
			if (strcmp(ent->d_name, ".") == 0 || strcmp(ent->d_name, "..") == 0)
				continue;

			if (strlen("./data")+strlen(ent->d_name)+2 > sizeof(path))
				fprintf(stderr, "name %s %s too long\n", "./data", ent->d_name);
			else
			{
				sprintf(path, "%s/%s", "./data", ent->d_name);

				load_data(path, db);
			}
			
		}

		closedir(dir);
	}
	else
	{
		perror("read file error!");
		return EXIT_FAILURE;
	}

	cudaEventRecord(end, 0);
	cudaEventSynchronize(end);

	cudaEventElapsedTime(&time, start, end);

	cout << "Read files elapsing " << time << " milli-seconds \n";

	// order all points by key
	thrust::sort_by_key(db.col_id_vec.begin(), db.col_id_vec.end(), db.col_lon_vec.begin());
	thrust::sort_by_key(db.col_id_vec.begin(), db.col_id_vec.end(), db.col_lat_vec.begin());
	thrust::sort_by_key(db.col_id_vec.begin(), db.col_id_vec.end(), db.col_time_vec.begin());

	//typedef thrust::tuple<int, double, double, long> out_tuple;
	thrust::device_vector<int> id_vec(5*db.size);
	thrust::device_vector<double> lon_vec(5*db.size);
	thrust::device_vector<double> lat_vec(5*db.size);
	thrust::device_vector<long> time_vec(5*db.size);

	cudaEventRecord(start, 0);

	line_interpolate llt(thrust::raw_pointer_cast(db.col_id_vec.data()), thrust::raw_pointer_cast(db.col_lon_vec.data()),
				 thrust::raw_pointer_cast(db.col_lat_vec.data()), thrust::raw_pointer_cast(db.col_time_vec.data()),
				 thrust::raw_pointer_cast(id_vec.data()), thrust::raw_pointer_cast(lon_vec.data()), thrust::raw_pointer_cast(lat_vec.data()), 
					thrust::raw_pointer_cast(time_vec.data()));
	thrust::for_each(thrust::counting_iterator<int>(0),thrust::counting_iterator<int>(db.size-1), llt);
	/*thrust::transform(thrust::make_zip_iterator(thrust::make_tuple(db.col_id_vec.begin(), 
				db.col_lon_vec.begin(), db.col_lat_vec.begin(), db.col_time_vec.begin())),
			thrust::make_zip_iterator(thrust::make_tuple(db.col_id_vec.end()-1, 
				db.col_lon_vec.end()-1, db.col_lat_vec.end()-1, db.col_time_vec.end()-1)),
			thrust::make_zip_iterator(thrust::make_tuple(db.col_id_vec.begin()+1, 
				db.col_lon_vec.begin()+1, db.col_lat_vec.begin()+1, db.col_time_vec.begin()+1)),
			out_vec.begin(), line_interpolate());
	
	*/
	//thrust::copy_n(lon_vec.begin(), 300, std::ostream_iterator<double>(std::cout, ","));
	
	cudaEventRecord(end, 0);
	cudaEventSynchronize(end);

	cudaEventElapsedTime(&time, start, end);

	cout << "interpolation elapsing " << time << " milli-seconds \n";

	cudaEventRecord(start, 0);

	auto new_end_id = thrust::remove_if(id_vec.begin(), id_vec.end(), thrust::placeholders::_1 == 0);
	thrust::remove_if(lon_vec.begin(), lon_vec.end(), thrust::placeholders::_1 == 0);
	thrust::remove_if(lat_vec.begin(), lat_vec.end(), thrust::placeholders::_1 == 0);
	thrust::remove_if(time_vec.begin(), time_vec.end(), thrust::placeholders::_1 == 0);

	const unsigned int length = new_end_id - id_vec.begin();

	thrust::device_vector<int> win_id_vec(length);
	thrust::device_vector<double> win_lon_vec(length);
	thrust::device_vector<double> win_lat_vec(length);
	thrust::device_vector<long> win_time_vec(length);

	/*slide sld(thrust::raw_pointer_cast(id_vec.data()), thrust::raw_pointer_cast(lon_vec.data()), thrust::raw_pointer_cast(lat_vec.data()),
		 thrust::raw_pointer_cast(time_vec.data()), thrust::raw_pointer_cast(win_id_vec.data()), thrust::raw_pointer_cast(win_lon_vec.data()),
			 thrust::raw_pointer_cast(win_lat_vec.data()), thrust::raw_pointer_cast(win_time_vec.data()), 8);
	thrust::for_each(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(length-8), sld);
	*/

	// copy data into sliding window vectors
	thrust::copy_n(id_vec.begin(), length, win_id_vec.begin());
	thrust::copy_n(lon_vec.begin(), length, win_lon_vec.begin());
	thrust::copy_n(lat_vec.begin(), length, win_lat_vec.begin());
	thrust::copy_n(time_vec.begin(), length, win_time_vec.begin());

	// release former vectors
	id_vec.clear(); lon_vec.clear(); lat_vec.clear(); time_vec.clear(); 
	id_vec.shrink_to_fit(); lon_vec.shrink_to_fit(); lat_vec.shrink_to_fit(); time_vec.shrink_to_fit();
	//thrust::device_vector<int>().swap(id_vec);
	//thrust::device_vector<double>().swap(lon_vec);
	//thrust::device_vector<double>().swap(lat_vec);
	//thrust::device_vector<long>().swap(time_vec);

	cudaEventRecord(end, 0);
	cudaEventSynchronize(end);

	cudaEventElapsedTime(&time, start, end);

	cout << "sliding elapsing " << time << " milli-seconds \n";

	//thrust::copy_n(win_time_vec.begin(), 500, std::ostream_iterator<long>(std::cout, ","));

	/*------------- start to do kNN search -----------------*/
	cudaEventRecord(start, 0);

	auto ptr_max_lon = thrust::max_element(db.col_lon_vec.begin(), db.col_lon_vec.end());
	auto ptr_min_lon = thrust::min_element(db.col_lon_vec.begin(), db.col_lon_vec.end());
	auto ptr_max_lat = thrust::max_element(db.col_lat_vec.begin(), db.col_lat_vec.end());
	auto ptr_min_lat = thrust::min_element(db.col_lat_vec.begin(), db.col_lat_vec.end());

	init_hash_function(200, 16, 1.0, 2.0, 1);
	int *hash_data = proj_points(200, 8, length, 8, win_id_vec, win_lon_vec, win_lat_vec, 
		*ptr_min_lon, *ptr_max_lon, *ptr_min_lat, *ptr_max_lat);

	cudaEventRecord(end, 0);
	cudaEventSynchronize(end);

	cudaEventElapsedTime(&time, start, end);

	cout << "project elapsing " << time << " milli-seconds \n";

	vector<vector<int> > queries;
	//vector<vector<int> > data;
	GPUGenie::inv_table *table = new GPUGenie::inv_table[1];
	table[0].set_table_index(0);
	table[0].set_total_num_of_table(1);

	//initialize configuration
	GPUGenie::GPUGenie_Config config;
	
	config.dim = 16;
	config.count_threshold = 14;
	config.num_of_topk = 5;
	config.hashtable_size = 14*config.num_of_topk*1.5;
	config.query_radius = 0;
	config.use_device = 0;
	config.use_adaptive_range = false;
	config.selectivity = 0.0f;

	config.query_points = &queries;
	//config.data_points = &data;
	
	config.use_load_balance = false;
	config.posting_list_max_length = 6400;
	config.multiplier = 1.5f;
	config.use_multirange = false;
	
	config.data_type = 0;
	config.search_type = 0;
	config.max_data_size = 0;

	config.num_of_queries = 3;

	// create inverted list
	to_inv_list(config, table, hash_data, length/8, 200);

	// prepare query input data
	//GPUGenie::read_file(queries, "test.csv", config.num_of_queries);
	double lon_query[24] = {	116.35092,116.34962,116.31765,116.32353,116.33932,116.33975,116.33706,116.33693,
					116.33583,116.33637,116.33985,116.34076,116.34052,116.37568,116.37568,116.4129,
					116.41215,116.41156,116.41119,116.40309,116.39041,116.38461,116.37348,116.36722
			    };
	double lat_query[24] = {	39.63793,39.63555,39.64407,39.69038,39.72047,39.72237,39.73823,39.76157,
					39.76158,39.76861,39.78241,39.79948,39.84915,39.86949,39.86949,39.89116,
					39.90378,39.91736,39.92837,39.93218,39.9321,39.92118,39.92994,39.92632
			    };
	
	thrust::device_vector<int> query_id_vec(24);
	thrust::device_vector<double> query_lon_vec(lon_query, lon_query+24);
	thrust::device_vector<double> query_lat_vec(lat_query, lat_query+24);
	int *query_hash_data = proj_points(200, 8, 24, 8, query_id_vec, query_lon_vec, query_lat_vec, 116.31765, 116.4129, 39.63555, 39.93218);
	for (int i = 0; i < 24; i++)
	{
		vector<int> row;
		for (int j = 0; j < 200; j++) row.push_back(query_hash_data[i*200+j]);
		queries.push_back(row);
	}
	
	// knn search
	vector<int> result;
	vector<int> result_count;
	GPUGenie::knn_search_after_preprocess(config, table, result, result_count);
	
	delete [] table;
	//getchar();
}
	