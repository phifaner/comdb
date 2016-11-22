#include <thrust/device_vector.h>

#include "comdb.h"

struct heatmap
{
    double left_lon, left_lat, right_lon, right_lat;        // heatmap area

    thrust::device_vector<double> bbox_lat, bbox_lon;              // bounding box of each cell

    int num_x, num_y;
    
    heatmap(int x, int y, double lf_lon, double lf_lat, double rt_lon, double rt_lat);

    int divide(int num_x, int num_y);                       // divide the space into num_x * num_y cells

    int* query_count(comdb *db);                             // query trajectories in each cell

    int* query_heatmap(comdb *db);
};

struct divide_map
{
    double left_lon, left_lat, right_lon, right_lat;        // heatmap area

    double *bbx_lon, *bbx_lat;                            // bounding box latitude and longitude

    int num_x, num_y;

    divide_map(const double lf_lon, const double lf_lat,
            const double rt_lon, const double rt_lat, double *_bbx_lon, double *_bbx_lat,
            const int x, const int y) : left_lon(lf_lon), left_lat(lf_lat), bbx_lon(_bbx_lon), bbx_lat(_bbx_lat),
            right_lon(rt_lon), right_lat(rt_lat), num_x(x), num_y(y) {}

    template <typename T>
    __host__ __device__
    void operator() (const T& i)
    {
        // computing cell's size both in longitude and latitude
        double delta_lon = (right_lon - left_lon) / num_x;
        double delta_lat = (left_lat - right_lat) / num_y;

        double i_left_lat = left_lat - ((int)(i/num_x)) * delta_lat;
        double i_left_lon = left_lon + ((int)(i%num_x)) * delta_lon;
        double i_right_lat = i_left_lat - delta_lat;
        double i_right_lon = i_left_lon + delta_lon;
	
	//printf("===========%d ", i);
	
        bbx_lat[i*2] = i_left_lat;
        bbx_lat[i*2+1] = i_right_lat;
        bbx_lon[i*2] = i_left_lon;
        bbx_lon[i*2+1] = i_right_lon;
	//printf("===========%lf", i_left_lat);

    }
};

struct query_cell
{
    size_t size;
    double *lon_vec, *lat_vec;

    query_cell(size_t _s, double *_lon, double *_lat) : size(_s), lon_vec(_lon), lat_vec(_lat) {}

    template <typename Tuple>
    __host__ __device__
    void operator()(const Tuple t) 
    {
	double lon = thrust::get<0>(t);
	double lat = thrust::get<1>(t);
	int id = thrust::get<2>(t);
	
	for (int i = 0; i < size; i++)
    	{
	    double lf_lon = lon_vec[2*i];
	    double lf_lat = lat_vec[2*i];
	    double rt_lon = lon_vec[2*i+1];
	    double rt_lat = lat_vec[2*i+1];
	    
	    if (lon > lf_lon && lon < rt_lon && lat < lf_lat && lat > rt_lat)
	    {	
		thrust::get<3>(t) = id;
		thrust::get<4>(t) = i;
	    }
	}
	 
    }
};

struct make_pair_functor
{
    template<typename T1, typename T2>
    __host__ __device__
    thrust::pair<T1, T2> operator() (const T1 &x, const T2 &y)
    {
	return thrust::make_pair(x, y);
    }
};

struct is_num_odd
{
    template <typename T>
    __host__ __device__
    bool operator() (const T& i)
    {
        return (i % 2) == 1;
    }
};

struct is_num_even
{
    template <typename T>
    __host__ __device__
    bool operator() (const T& i)
    {
        return (i % 2) == 0;
    }
};

struct is_valuable
{
	template <typename T>
	__host__ __device__
	bool operator() (const T& i, const T& j)
	{	
		// compare two neighbor id, if they both > 0 and they have distinct number, return 1
		if (i != j)
		{
			if (i > 0 || j > 0) return 1;
		}
		 return 0;
	}
};

struct is_zero
{
	__host__ __device__
	bool operator()(const int& x)
	{
		return x > 0;
	}
};

struct equal_plus
{
	int *keys;
	int *values;
	int *counts;
	
	equal_plus(int *k, int *v, int *c):keys(k), values(v), counts(c) {}

	template <typename T>
	__host__ __device__
	void operator() (const T& i)
	{
		if (i == 0) counts[keys[i]] = 1;
		else 
		{
			//if (keys[i] == 36) printf("%d ,", values[i]);
			if (keys[i] > 0 && keys[i] != keys[i-1])
			{
				atomicAdd(counts+keys[i], 1);
			}
			else
			{
				if (values[i] > 0 && values[i] != values[i-1])	
					atomicAdd(counts+keys[i], 1);
			}

		
		}
	}
};
