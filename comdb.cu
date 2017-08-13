#include <thrust/for_each.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/remove.h>

#include "comdb.h"
#include "parser.h"
#include "hash_table.h"

#include <sys/mman.h>
#include <unistd.h>
#include <fcntl.h>
#include <time.h>

void comdb::select_by_id(int id)
{
    id_equal ieq(id);
   
    size_t _size = col_id_vec.size(); 	
    res_lat_vec.resize(_size);
    res_id_vec.resize(_size);
    res_lon_vec.resize(_size);
    res_time_vec.resize(_size);

    thrust::for_each(
            thrust::make_zip_iterator(thrust::make_tuple(col_id_vec.begin(), col_lat_vec.begin(),
		        col_lon_vec.begin(), col_time_vec.begin(),  res_id_vec.begin(), res_lat_vec.begin(), 
                    res_lon_vec.begin(), res_time_vec.begin())),
            thrust::make_zip_iterator(thrust::make_tuple(col_id_vec.end(), col_lat_vec.end(), 
		        col_lon_vec.end(), col_time_vec.end(), res_id_vec.end(), res_lat_vec.end(), 
                    res_lon_vec.end(), res_time_vec.end())),
            ieq);

 //	thrust::copy(res_id_vec.begin(), res_id_vec.end(), std::ostream_iterator<int>(std::cout, ","));
}

size_t comdb::select_by_id_array(int *id_array, unsigned long len, long *t)
{
        kernel_id_array kia(id_array, len, t);

        size_t _size = col_id_vec.size();   
        res_lat_vec.resize(_size);
        res_id_vec.resize(_size);
        res_lon_vec.resize(_size);
        res_time_vec.resize(_size);

        thrust::for_each(
            thrust::make_zip_iterator(thrust::make_tuple(col_id_vec.begin(), col_lat_vec.begin(),
                col_lon_vec.begin(), col_time_vec.begin(),  res_id_vec.begin(), res_lat_vec.begin(), 
                    res_lon_vec.begin(), res_time_vec.begin())),
            thrust::make_zip_iterator(thrust::make_tuple(col_id_vec.end(), col_lat_vec.end(), 
                col_lon_vec.end(), col_time_vec.end(), res_id_vec.end(), res_lat_vec.end(), 
                    res_lon_vec.end(), res_time_vec.end())),
            kia);

        auto end_ptr = thrust::remove_if(res_id_vec.begin(), res_id_vec.end(), thrust::placeholders::_1 == -1);
        thrust::remove_if(res_lat_vec.begin(), res_lat_vec.end(), thrust::placeholders::_1 == -1);
        thrust::remove_if(res_lon_vec.begin(), res_lon_vec.end(), thrust::placeholders::_1 == -1);
        thrust::remove_if(res_time_vec.begin(), res_time_vec.end(), thrust::placeholders::_1 == -1);
        
        return (end_ptr - res_id_vec.begin());

        // thrust::copy_n(res_id_vec.begin(), length, std::ostream_iterator<double>(std::cout, ","));
}


int comdb::select_all_id(int *result)
{
	thrust::device_vector<int> id_vec(size);
	thrust::stable_sort(col_id_vec.begin(), col_id_vec.end());
	
	// if not two neighbor id equal, put in
	get_all_id gad(thrust::raw_pointer_cast(id_vec.data()), thrust::raw_pointer_cast(col_id_vec.data()));
	thrust::for_each(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(size), gad);	

	//thrust::copy_n(id_vec.begin(), id_vec.end(), std::ostream_iterator<int>(std::cout, ","));

	auto end_ptr = thrust::remove_if(id_vec.begin(), id_vec.end(), thrust::placeholders::_1 == -1);
	
	int length = end_ptr - id_vec.begin();
	thrust::copy_n(id_vec.begin(), length, result);
	
	//thrust::copy_n(result, length, std::ostream_iterator<int>(std::cout, ","));

	return length;
}

int comdb::select_by_time(const char *start, const char *end)
{
	// parse time into long value
	struct tm tm_start, tm_end;
	time_t t_start, t_end;
	
	if (strptime(start, "%Y-%m-%d %H:%M:%S", &tm_start) == NULL) 
	{
		perror("parse date error!");
		return EXIT_FAILURE;
	}
	
	if (strptime(end, "%Y-%m-%d %H:%M:%S", &tm_end) == NULL)
	{
		perror("parse date error!");
		return EXIT_FAILURE;
	}
	
	// we are at GMT-8, so add 8 hours
	t_start = mktime(&tm_start) + 8 * 3600;
	t_end = mktime(&tm_end) + 8 * 3600;

	time_between tbn(t_start, t_end);
	
	size_t _size = col_time_vec.size();
	res_lat_vec.resize(_size);

	thrust::for_each(
		thrust::make_zip_iterator(thrust::make_tuple(col_time_vec.begin(), col_lat_vec.begin(), res_lat_vec.begin())),
	    	thrust::make_zip_iterator(thrust::make_tuple(col_time_vec.end(), col_lat_vec.end(), res_lat_vec.end())),
		tbn);
	return col_time_vec.end() - col_time_vec.begin();
}


int comdb::select_by_space(double top_left_lon, double top_left_lat, double bottom_right_lon, double bottom_right_lat)	
{
 
	space_between sbn(top_left_lon, top_left_lat, bottom_right_lon, bottom_right_lat);
	
	// resize result vector equal to the size of data columns
	size_t _size = col_lon_vec.size();
	res_id_vec.resize(_size);	
	thrust::fill(res_id_vec.begin(), res_id_vec.end(), -1);

	thrust::for_each(
		thrust::make_zip_iterator(thrust::make_tuple(col_lon_vec.begin(), col_lat_vec.begin(), 
			col_id_vec.begin(), res_id_vec.begin())),
		thrust::make_zip_iterator(thrust::make_tuple(col_lon_vec.end(), col_lat_vec.end(), 
			col_id_vec.end(), res_id_vec.end())),
		sbn
	); 	

	//	std::cout << "----------- line 257" << std::endl;
	

  	//thrust::copy(res_id_vec.begin(), res_id_vec.end(), std::ostream_iterator<int>(std::cout, "\n"));

	size = res_id_vec.end() - res_id_vec.begin();
	return size;
}

int comdb::select_by_space_time(double top_left_lon, double top_left_lat, double bottom_right_lon, 
				double bottom_right_lat, long start, long end)
{
	space_time_between stb(top_left_lon, top_left_lat, bottom_right_lon, bottom_right_lat, start, end);
	
	size_t _size = col_lon_vec.size();
	res_id_vec.resize(_size);

	thrust::for_each(
		thrust::make_zip_iterator(thrust::make_tuple(col_lon_vec.begin(), col_lat_vec.begin(), 
			col_time_vec.begin(), col_id_vec.begin(), res_id_vec.begin())),
		thrust::make_zip_iterator(thrust::make_tuple(col_lon_vec.end(), col_lat_vec.end(),
			col_time_vec.end(), col_id_vec.end(), res_id_vec.end())),
		stb
	);

	return res_id_vec.end() - res_id_vec.begin();
}
