#ifndef _COMDB_H
#define _COMDB_H

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>

#include "trajectory.h"

struct comdb
{
    // store trajectory column's values
    thrust::device_vector<int>      col_id_vec;
    thrust::device_vector<double>   col_lon_vec;
    thrust::device_vector<double>   col_lat_vec;
    thrust::device_vector<long>     col_time_vec;
    thrust::device_vector<int>      col_state_vec;
    thrust::device_vector<int>      col_speed_vec;
    thrust::device_vector<int>      col_dir_vec;

    // store query result
    thrust::device_vector<int>      res_id_vec;
    thrust::device_vector<double>   res_lon_vec;
    thrust::device_vector<double>   res_lat_vec;
    thrust::device_vector<long>     res_time_vec;
    thrust::device_vector<int>      res_state_vec;
    thrust::device_vector<int>      res_speed_vec;
    thrust::device_vector<int>      res_dir_vec;



    size_t size;                    // table size

    /* get all trajectory ids*/
    int select_all_id(int *result);    		

    /* query by id */
    void select_by_id(int id);   

    /* query by multiple ids */
    size_t select_by_id_array(int *id_array, unsigned long len, long *t);      		     	     		

    /* query by time interval */
    int select_by_time(const char *start, const char *end);
    
	/* query id by a spatial window*/
    int select_by_space(double top_left_lon, double top_left_lat, 
		double bottom_right_lon, double bottom_right_lat);		
    
    /* query id by a space-time window */
    int select_by_space_time(double top_left_lon, double top_left_lat,
		double bottom_right_lon, double bottom_right_lat,		
		long start, long end);

};

/* GPU code for select_by_id */
struct id_equal
{
    int id;

    __host__ __device__
    id_equal(int _id) : id(_id) { }

    // read each line of id, if id matches return the corresponding column value
    // id, trajectory, and result form a Tuple
    template <typename Tuple>
   __host__  __device__
    void operator() (const Tuple t)
    {
        int data = thrust::get<0>(t);
        double lat = thrust::get<1>(t);
		double lon = thrust::get<2>(t);
		long ts = thrust::get<3>(t);
        
		if (data == id) 
		{
			thrust::get<4>(t) = id;
			thrust::get<5>(t) = lat;
			thrust::get<6>(t) = lon;
			thrust::get<7>(t) = ts;
		}
	        else   
		{
			thrust::get<4>(t) = -1;
			thrust::get<5>(t) = -1;
			thrust::get<6>(t) = -1;
			thrust::get<7>(t) = 0;
		}

    }
};


/* kernel for select_by_id_array */
struct kernel_id_array
{
	int *id_array;
	unsigned long len;
	// int id;
	long *T;		// start time and end time

	__host__ __device__
	kernel_id_array(int *_ids, unsigned long _len, long *_t) : id_array(_ids), len(_len), T(_t) { }
	// kernel_id_array(int _id, long *_t) : id(_id), T(_t) {}

	template <typename Tuple>
	__host__ __device__
	void operator() (const Tuple& t)
	{
		int data = thrust::get<0>(t);
        double lat = thrust::get<1>(t);
		double lon = thrust::get<2>(t);
		long ts = thrust::get<3>(t);


		for (int i = 0; i < len; i++)	// why it doesn't work when using while loop ??
		{
			int id = id_array[i];

			if ( data != id || !(ts > T[0] && ts < T[1]) )
			{
				thrust::get<4>(t) = -1;
				thrust::get<5>(t) = -1;
				thrust::get<6>(t) = -1;
				thrust::get<7>(t) = 0;
			}
			else
			{
				thrust::get<4>(t) = id;
				thrust::get<5>(t) = lat;
				thrust::get<6>(t) = lon;
				thrust::get<7>(t) = ts;

				break;	// find the matched point, end the loop
			}
		}
	}

};

/* kernel for select_all_id */
struct get_all_id
{
	int *all_id, *in_data;

	get_all_id(int *ids, int *in) : all_id(ids), in_data(in) {}

	__host__ __device__
	void operator() (const int& i)
	{
        /* keep the first id */
		if (i == 0) all_id[i] = in_data[i];

        /* continuous id not equal */
        if (i > 0 && in_data[i] > 0 && in_data[i] != in_data[i-1])
		{
			all_id[i] = in_data[i];		
		}
		else 
		{
			all_id[i] = -1;
		}
	}
};

/* kernel for select_by_time */
struct time_between
{
	long start, end;
	
	__host__ __device__
	time_between(long s, long e) : start(s), end(e) {}
	
	
	__host__ __device__ 
	int cu_strcmp(const char *str_a, const char *str_b, unsigned int len = 256)
	{
		int match = 0;
		unsigned i = 0;
		unsigned done = 0;

		while ((i < len) && (match == 0) && !done)
		{	
			// meet to the end of a string
			if ((str_a[i] == 0) || (str_b[i] == 0)) done = 1;
			else if (str_a[i] != str_b[i])
			{
				match = i + 1;
				if (((int)str_a[i] - (int)str_b[i]) < 0)
					match = 0 - (i + 1);
			}
			
			i++;
		}
		
		return match;
	}

	template <typename Tuple>
	__host__ __device__
	void operator() (const Tuple t)
	{
		long data = thrust::get<0>(t);
		double lat = thrust::get<1>(t);

		if ( data >= start && data <= end)
			thrust::get<2>(t) = lat;							
		else
			thrust::get<2>(t) = -1;
	}

};


/* kernel for select_by_space */
struct space_between
{
	double left_lon, left_lat, right_lon, right_lat;
	
	__host__ __device__
	space_between(double _lf_lon, double _lf_lat, double _rt_lon, double _rt_lat):
		left_lon(_lf_lon), left_lat(_lf_lat), right_lon(_rt_lon), right_lat(_rt_lat) {}
	
	template <typename Tuple>
	__host__ __device__	
	void operator() (const Tuple t)
	{
		//printf("----------------------\n");
		double lon = thrust::get<0>(t);
		double lat = thrust::get<1>(t);
		long id = thrust::get<2>(t);
		//printf("=============%ld\n", id);
		if (lon < right_lon && lon > left_lon && lat > right_lat && lat < left_lat)
			thrust::get<3>(t) = id;
		else
			thrust::get<3>(t) = -1;
	}
};

/* kernel for select_by_space_time */
struct space_time_between
{
	double left_lon, left_lat, right_lon, right_lat;
	long start, end;

	__host__ __device__
	space_time_between(double _lf_lon, double _lf_lat, double _rt_lon, 
            double _rt_lat, long _start, long _end):
		left_lon(_lf_lon), left_lat(_lf_lat), right_lon(_rt_lon), 
            right_lat(_rt_lat), start(_start), end(_end) {}

	template <typename Tuple>
	__host__ __device__
	void operator() (const Tuple t)
	{
		double lon = thrust::get<0>(t);
		double lat = thrust::get<1>(t);
		long time = thrust::get<2>(t);
		long id = thrust::get<3>(t);
		

		if (lon < right_lon && lon > left_lon && lat > right_lat 
                    && lat < left_lat && time > start && time < end)
			thrust::get<4>(t) = id;
		else 
			thrust::get<4>(t) = -1;
	}
};


struct points_to_traj
{
	int		tid;
	int		*tid_data;
	double 		*lon_data, *lon_res;
	double 		*lat_data, *lat_res;
	long   		*ts_data, *ts_res;

	points_to_traj(int _tid, int *_tids, double *_lon, double *_lat, long *_ts, double *_res_lon, double *_res_lat, long *_res_ts) : tid(_tid), tid_data(_tids), lon_data(_lon), 
				lat_data(_lat), ts_data(_ts), lon_res(_res_lon), lat_res(_res_lat), ts_res(_res_ts) {}

	__host__ __device__
	void operator()(const int& i) 
	{
		//if (tid_data[i] != -1)	printf("--------------%d\n line 217", tid_data[i]);
		if (tid_data[i] == tid)
		{	
			lon_res[i] = lon_data[i];	
			//printf("----------pt-- to traj %ld\n", ts_data[i]);
			lat_res[i] = lat_data[i];
			ts_res[i] = ts_data[i];
		}
	}
};

#endif
