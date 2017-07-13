#include <thrust/device_vector.h>

typedef struct
{
	size_t 	length;
	double* latitude;
	double* longitude;
	long*	ts;
} trajectory;

typedef struct
{
	double latitude;
	double longitude;
	long   ts;
} tpoint;

typedef struct
{
	size_t length;
	tpoint *buffer;	
} swindow;


struct slide
{
	size_t		num;
	swindow 	*swin;
	trajectory	traj;

	slide(size_t _num, swindow *_swin, trajectory _traj) : num(_num), swin(_swin), traj(_traj) {}

	//template <typename T>
	__host__ __device__
	void operator() (const unsigned int i)
	{
		//thrust::for_each(thrust::counting_iterator<unsigned int>(0), thrust::counting_iterator<unsigned int>(traj.length), [=]__device__(const int& j)
		{
			//cudaMalloc((void**)&swin, sizeof(size_t)+sizeof(tpoint)*num);
			swin[i].buffer = new tpoint[num];			

			for (int k = 0; k < num; k++)
			{
				swin[i].buffer[k].longitude = traj.longitude[i+k];
				swin[i].buffer[k].latitude = traj.latitude[i+k];
				swin[i].buffer[k].ts = traj.ts[i+k];
			}
		};
		
	}
};

int main()
{
	double LAT[15] = {28.289, 28.287, 28.286, 28.285, 28.283, 28.284, 28.287, 28.282, 28.284, 28.286, 28.281, 28.286, 28.289, 28.279, 28.278};
	double LON[15] = {121.11, 121.23, 121.20, 121.25, 121.22, 121.12, 121.02, 121.03, 121.03, 121.22, 121.21, 121.26, 121.12, 121.11, 121.20};
	long  TS[15] = {12638782800, 12638782900, 12638783000, 12638783100, 12638783200, 12638783300, 12638783400, 12638783500, 12638783600, 12638783700,
			12638783800, 12638783900, 12638784000, 12638784100, 12638784200};

	thrust::device_vector<double> lat_vec(LAT, LAT+15);
	thrust::device_vector<double> lon_vec(LON, LON+15);
	thrust::device_vector<long> ts_vec(TS, TS+15);

	trajectory traj;
	traj.latitude = thrust::raw_pointer_cast(lat_vec.data());
	traj.longitude = thrust::raw_pointer_cast(lon_vec.data());
	traj.ts = thrust::raw_pointer_cast(ts_vec.data());
	traj.length = 15;

	thrust::device_vector<swindow> win_vec(15);
	
	slide sld(8, thrust::raw_pointer_cast(win_vec.data()), traj);

	thrust::for_each(thrust::counting_iterator<unsigned int>(0), thrust::counting_iterator<unsigned int>(traj.length), sld);
	
}
