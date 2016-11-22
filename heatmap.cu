#include <thrust/copy.h>
#include <thrust/sort.h>
#include <thrust/inner_product.h>
#include <thrust/replace.h>
#include <thrust/fill.h>
#include <thrust/host_vector.h>
#include <thrust/set_operations.h>
#include <thrust/system/tbb/execution_policy.h>

#include "heatmap.h"


heatmap::heatmap(int x, int y, double lf_ln, double lf_lt, double rt_ln, double rt_lt) : num_x(x), num_y(y)
{
    left_lon = lf_ln;
    left_lat = lf_lt;
    right_lon = rt_ln;
    right_lat = rt_lt;

    std::vector<double> a(num_x*num_y*2);
    bbox_lat = a;
    //fill(bbox_lat.begin(), bbox_lat.end(), 0);
    bbox_lon = a;
    //thrust::fill(bbox_lon.begin(), bbox_lon.end(), 0);
}

int heatmap::divide(int num_x, int num_y)
{

	//std::cout << "---------------" << bbox_lat.size() << std::endl;
    thrust::counting_iterator<unsigned int> begin(0);
    thrust::counting_iterator<unsigned int> end(num_x*num_y);
    divide_map divide(left_lon, left_lat, right_lon, right_lat, 
            (double*)thrust::raw_pointer_cast(bbox_lon.data()),
            (double*)thrust::raw_pointer_cast(bbox_lat.data()),
            num_x, num_y);

	//std::cout << "iiiiiiiiiiiiiiiiiiii" << std::endl;
    thrust::for_each(begin, end, divide);

    return 1;
}

int* heatmap::query_count(comdb *p_db)
{
    thrust::host_vector<int> count_vec(num_x*num_y, 0);
    
    /*// store four value of each bounding box
    thrust::device_vector<double> left_lon_vec(num_x*num_y);
    thrust::device_vector<double> left_lat_vec(num_x*num_y);
    thrust::device_vector<double> right_lon_vec(num_x*num_y);
    thrust::device_vector<double> right_lat_vec(num_x*num_y);

    // only copy half amount of values
    thrust::device_vector<int> stencil_odd(num_x*num_y);
    thrust::fill(stencil_odd.begin(), stencil_odd.end(), 1);
    is_num_odd odd;
    is_num_even even;
    thrust::replace_if(stencil_odd.begin(), stencil_odd.end(), even, 0);
    
    thrust::device_vector<int> stencil_even(num_x*num_y);
    thrust::fill(stencil_even.begin(), stencil_even.end(), 1);
    thrust::replace_if(stencil_even.begin(), stencil_even.end(), odd, 0);

    thrust::copy_if(bbox_lon.begin(), bbox_lon.end(), stencil_odd.begin(), left_lon_vec.begin(), is_num_odd());
    thrust::copy_if(bbox_lon.begin(), bbox_lon.end(), stencil_even.begin(), right_lon_vec.begin(), is_num_even());
    thrust::copy_if(bbox_lat.begin(), bbox_lat.end(), stencil_odd.begin(), left_lat_vec.begin(), is_num_odd());
    thrust::copy_if(bbox_lat.begin(), bbox_lat.end(), stencil_even.begin(), right_lat_vec.begin(), is_num_even());
    
    // query trajectories in each cell	
    query_cell qc(p_db);
    thrust::for_each(
            thrust::make_zip_iterator(thrust::make_tuple(left_lon_vec.begin(), left_lat_vec.begin(), 
                    right_lon_vec.begin(), right_lat_vec.begin(), id_vec.begin())),
            thrust::make_zip_iterator(thrust::make_tuple(left_lon_vec.end(), left_lat_vec.end(), 
                    right_lon_vec.end(), right_lat_vec.end(), id_vec.end())),
            qc);

    // count trajectory id
   // thrust::sort(id_vec.begin(), id_vec.end());
	std::cout << id_vec.size();
  //  for (int i = 0; i < id_vec.size(); i++)
    {
	//thrust::sort((int*)thrust::raw_pointer_cast(id_vec[i]), id_vec[i]+p_db->size);
    

    	//size_t count = thrust::inner_product(id_vec[0], id_vec[0]+p_db->size-1,
//            	id_vec[0]+1, 0, thrust::plus<int>(), thrust::not_equal_to<int>());
	
//	vstd::cout << "-----------" << count << "\n";
    }

    return 1;*/
	cudaEvent_t start, end;
	float time;

    for (int i = 0; i < num_x*num_y; i++)
    {
	double lf_lon = bbox_lon[2*i];
	double lf_lat = bbox_lat[2*i];
	double rt_lon = bbox_lon[2*i+1];
	double rt_lat = bbox_lat[2*i+1];
	
	//thrust::device_vector<int> temp_vec()

	//std::cout << lf_lon << "," << lf_lat << "," << rt_lon << "," << rt_lat << std::endl;	
	cudaEventCreate(&start);
	cudaEventCreate(&end);
	
	cudaEventRecord(start, 0);

	p_db->select_by_space(lf_lon, lf_lat, rt_lon, rt_lat);

	thrust::device_vector<int> temp_vec(p_db->res_id_vec.size());	
	
	//remove data of no use and shrink to exactly fit output
	/*thrust::device_vector<int>::iterator new_end = 
		thrust::remove_if(p_db->res_id_vec.begin(), p_db->res_id_vec.end(), is_zero());
	p_db->res_id_vec.erase(new_end, p_db->res_id_vec.end());
	*/
	//thrust::stable_sort(p_db->res_id_vec.begin(), p_db->res_id_vec.end());
	//thrust::set_intersection(p_db->res_id_vec.begin(), p_db->res_id_vec.end(),
	//			 p_db->res_id_vec.begin(), p_db->res_id_vec.end(), temp_vec.begin());

	size_t count = thrust::inner_product(p_db->res_id_vec.begin(), p_db->res_id_vec.end()-1,
            	p_db->res_id_vec.begin()+1, 0, thrust::plus<int>(), is_valuable());
	//if (i == 5)
	//	thrust::copy(temp_vec.begin(), temp_vec.end(), std::ostream_iterator<int>(std::cout, ","));		

	cudaEventRecord(end, 0);
	cudaEventSynchronize(end);

	cudaEventElapsedTime(&time, start, end);
	
	//std::cout << "--------------select ---------" << time << "\n";
	// counting
	//thrust::sort(p_db->res_id_vec.begin(), p_db->res_id_vec.end());
	
	/*count_vec[i] = thrust::count_if(p_db->res_id_vec.begin(), p_db->res_id_vec.end(), is_zero());
	*/
	//thrust::sort(p_db->res_id_vec.begin(), p_db->res_id_vec.end());

	/*if (i == 13)
	{	
		
		int k = 0;
		thrust::copy(p_db->res_id_vec.begin(), p_db->res_id_vec.end(),
			 std::ostream_iterator<int>(std::cout, ","));
		for (int j = 0; j < p_db->res_id_vec.size(); j++)
		{
			
			if (p_db->res_id_vec[j] > 0) 
			{
				std::cout << "========" << p_db->res_id_vec[j] << "\n";
				k++;
			}
		}
		std::cout << "--------------kkkkk" << k << std::endl;
	}
	//std::cout << "-----------" << count_vec[i] << "\n";*/
	
	//size_t count = thrust::inner_product(p_db->res_id_vec.begin(), p_db->res_id_vec.end()-1,
        //    	p_db->res_id_vec.begin()+1, 0, thrust::plus<int>(), is_valuable());

	//std::cout << "-------------" << count << std::endl;
    }
   
 	return (int*)thrust::raw_pointer_cast(count_vec.data());
    
}

int* heatmap::query_heatmap(comdb *p_db)
{
	int *values = new int[num_x*num_y];
	query_cell qc(num_x*num_y, (double*)thrust::raw_pointer_cast(bbox_lon.data()), 
			(double*)thrust::raw_pointer_cast(bbox_lat.data()));
    	
	// query trajectories in each cell
	thrust::device_vector<int> id_vec(p_db->size);	
    	thrust::device_vector<int> cell_vec(p_db->size);
	thrust::for_each(
            thrust::make_zip_iterator(thrust::make_tuple(p_db->col_lon_vec.begin(), p_db->col_lat_vec.begin(), 
                    	p_db->col_id_vec.begin(), id_vec.begin(), cell_vec.begin())),
            thrust::make_zip_iterator(thrust::make_tuple(p_db->col_lon_vec.end(), p_db->col_lat_vec.end(), 
			p_db->col_id_vec.end(), id_vec.end(), cell_vec.end())),
            qc);

	// import! sort keys and also sort values
	thrust::sort_by_key(cell_vec.begin(), cell_vec.end(), id_vec.begin());
	
	// count trajectory in each cell
	thrust::device_vector<int> count_vec(p_db->size);
	thrust::fill(count_vec.begin(), count_vec.end(), 0);

	equal_plus ep(thrust::raw_pointer_cast(cell_vec.data()), thrust::raw_pointer_cast(id_vec.data()),
			thrust::raw_pointer_cast(count_vec.data()));
 
	
	thrust::counting_iterator<unsigned int> begin(0);
	thrust::counting_iterator<unsigned int> end(p_db->size);
	thrust::for_each(begin, end, ep);
	
	thrust::copy_n(count_vec.begin(), num_x*num_y, values);
	thrust::copy_n(count_vec.begin(), num_x*num_y, std::ostream_iterator<int>(std::cout, ","));

	return values;

}

