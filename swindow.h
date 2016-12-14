#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>

#include "trajectory.h"

struct sliding_window
{
	size_t size;				// window size
	tpoint *buffer;				// store data in a window
	
	__host__ __device__
	sliding_window() {}
	
	__host__ __device__
	sliding_window(size_t _size) : size(_size)
	{
		thrust::device_vector<tpoint> vec(size);
		buffer = thrust::raw_pointer_cast(vec.data());
	}
};

// interpolate a trajectory s.t. it has a regular timestamp, such as 5 mins
void
interpolate(trajectory *in, trajectory *out);

// partition an interpolated trajectory with a specified window
void
slide_window(thrust::device_vector<trajectory> traj_vec, size_t num, thrust::device_vector<struct sliding_window> data);

// build LSH index for each slice of trajectory in a sliding window
void
lsh_index(struct sliding_window *data);



struct line_interpolate 
{
	int *tid_data, *out_tid_data;
	double *lon_data, *out_lon_data;
	double *lat_data, *out_lat_data;
	long *ts_data, *out_ts_data;
	
	line_interpolate(int *_tid, double *_lon, double *_lat, long *_ts, int *res_tid, double *_res_lon, double *_res_lat, long *_res_ts) 
		: tid_data(_tid), lon_data(_lon), lat_data(_lat), ts_data(_ts), out_tid_data(res_tid), out_lon_data(_res_lon), out_lat_data(_res_lat), out_ts_data(_res_ts) {}
	

	__host__ __device__
	void operator() (const unsigned int& i)
	{
		out_tid_data[5*i] = tid_data[i];
		out_lon_data[5*i] = lon_data[i];
		out_lat_data[5*i] = lat_data[i];
		out_ts_data[5*i] = ts_data[i];

		//if (lat_data[i] < 10) printf("line 43 ------------ %lf\n", lat_data[i]);
		if (tid_data[i] == tid_data[i+1])
		{
			long intervals = ts_data[i+1] - ts_data[i];
			
			//printf("----------time intervals--------%ld\n", intervals);
			// if missing data last longer than 5*300s, do nothing
			if (intervals > TIME_INTERVAL && intervals < 5*TIME_INTERVAL) 
				execute(tid_data[i+1], lon_data[i+1], lat_data[i+1], ts_data[i+1], 
						lon_data[i], lat_data[i], ts_data[i], out_tid_data, out_lon_data, out_lat_data, out_ts_data, i);
		}
		/*else // if < 300s, add a new point, if > 5*300 or == 300, miss the middle poionts
		{
			out_tid_data[5*i] = tid_data[i];
			out_lon_data[5*i] = lon_data[i];
			out_lat_data[5*i] = lat_data[i];
			out_ts_data[5*i] = ts_data[i];
		}*/ 
	}

	__host__ __device__
	void execute(int tid, double lon_1, double lat_1, long ts_1, double last_lon, double last_lat,
			 long last_ts, int *res_tid,  double *res_lon, double *res_lat, long *res_ts, int tix)
	{
		double min_lon = min(lon_1, last_lon);
		double max_lon = max(lon_1, last_lon);
		double min_lat = min(lat_1, last_lat);
		double max_lat = max(lat_1, last_lat);
		long min_ts = min(ts_1, last_ts);
		long max_ts = max(ts_1, last_ts);

		// the number of points need to add
		int num = (int)((max_ts - min_ts) / TIME_INTERVAL);
		
		// the step of longitude and latitude
		double step_lon = (max_lon-min_lon) / num;
		double step_lat = (max_lat-min_lat) / num;

		//cudaMalloc((void**)res_lon, num*sizeof(double));	
		//cudaMalloc((void**)res_lat, num*sizeof(double));
		//cudaMalloc((void**)res_ts, num*sizeof(long));

		// add points
		for (int k = 1; k < num+1; k++)
		{
			res_tid[5*tix+k] = tid;
			res_lon[5*tix+k] = min_lon + k*step_lon;
			//printf("line 90 ----------- %lf\n", res_lon[5*tix+k]);
			res_lat[5*tix+k] = max_lat - k*step_lat;
			res_ts[5*tix+k] = min_ts + k*TIME_INTERVAL;
			//if (res_lat[5*tix+k] < 10)
			//	printf("line 107 ----------- %lf\n", res_lat[5*tix+k]);
			
		}
		
		// record the last point, preparing for the next interpolation
		/*last_lon = res_lon[5*tix+num];
		last_lat = res_lat[5*tix+num];
		last_ts = res_ts[5*tix+num];*/
	}
};

struct slide
{
	int num;

	int *id_data;
	double *lon_data, *lat_data;
	long *ts_data;

	int *win_id_data;
	double *win_lon_data, *win_lat_data;
	long *win_time_data;
	
	slide(int *_tid, double *_lon, double *_lat, long *_ts, 
		int *win_tid, double *win_lon, double *win_lat, long *win_ts, int _num) 
		: num(_num), id_data(_tid), lon_data(_lon), lat_data(_lat), ts_data(_ts),
			win_id_data(win_tid), win_lon_data(win_lon), win_lat_data(win_lat), win_time_data(win_ts) {}

	template <typename T>
	__host__ __device__ 
	void operator() (const T& i)
	{
		// iterate each point of trajectory
		/*thrust::for_each(thrust::device, thrust::counting_iterator<unsigned int>(0), thrust::counting_iterator<unsigned int>(traj_ptr[i].length), [=]__device__(const int& j) 
		{
			//swindow_ptr[i].buffer = new tpoint[num];
			for (int k = 0; k < num; k++)
			{
				printf("line 148 longitude --------------%d\n", traj_ptr[i].length);
				win_lon_data[i] = lon_data[num*i];
				swindow_ptr[i*traj_ptr[i].length+j].buffer[k].latitude = traj_ptr[i].lat_data[j+k];
				swindow_ptr[i*traj_ptr[i].length+j].buffer[k].ts = traj_ptr[i].ts_data[j+k];
			}	
		});*/
	}
};

// delete (t =0 && t1 = t2) points
struct rremove
{
	double *lon_data;
	double *lat_data;
	long *ts_data;

	rremove(double *_lon, double *_lat, long *ts) : lon_data(_lon), lat_data(_lat), ts_data(ts) {}

	__host__ __device__
	bool operator() (const int i)
	{
		printf("------\n");
		if (i > 0 && ts_data[i] == ts_data[i-1])	
			return 0;
		return 1;
	}
};
