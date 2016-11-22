#include <thrust/for_each.h>
#include <iostream>
#include <iterator>

#include "swindow.h"

void
interpolate(trajectory *traj, trajectory *result)
{
	thrust::device_vector<trajectory> traj_vec(2);
	traj_vec[0] = *traj;
	thrust::device_vector<trajectory> res_traj_vec(2);
	res_traj_vec[0] = *result;
	line_interpolate linp((trajectory*)thrust::raw_pointer_cast(traj_vec.data()), (trajectory*)thrust::raw_pointer_cast(res_traj_vec.data()));

	thrust::counting_iterator<unsigned int> begin(0);
	thrust::counting_iterator<unsigned int> end(traj_vec.size());

	//thrust::copy(traj_vec.begin(), traj_vec.end(), std::ostream_iterator<trajectory>(std::cout, ","));
	thrust::for_each(begin, end, linp);

}

void
slide_window(thrust::device_vector<trajectory> traj_vec, size_t num, thrust::device_vector<struct sliding_window> data)
{
	struct sliding_window *swindow_ptr = thrust::raw_pointer_cast(data.data());
	trajectory *traj_ptr = thrust::raw_pointer_cast(traj_vec.data());

	// iterate each trajectory
	thrust::for_each(thrust::counting_iterator<unsigned int>(0), thrust::counting_iterator<unsigned int>(traj_vec.size()), slide(num, swindow_ptr, traj_ptr)); 
}
