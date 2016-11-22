#ifndef __TRAJECOTRY_H__
#define __TRAJECOTRY_H__

#define TIME_INTERVAL	300			// time interval between two points -- 300 seconds

typedef struct tpoint
{
	double 		longitude;
	double 		latitude;
	long 		ts;
} tpoint;

typedef struct trajectory
{
	int 		tid;			// unique id of a trajectory
	double* 	lon_data;		// locations of longitude
	double* 	lat_data;		// locations of latitude
	long*		ts_data;		// timestamp of each location
	size_t		length;			// length of a trajectory
} trajectory;

#endif
