#include <thrust/sort.h>
#include <thrust/extrema.h>

#include <dirent.h>

#include "comdb.h"
#include "parser.h"
#include "file_index.h"
#include "dbscan.h"
#include "quadtree.h"

using namespace std;

long long int totalAllocatedMemory = 0;

int main()
{
    cudaSetDevice(2);
                    
    comdb db;
                
    // record ellipsed time
    cudaEvent_t start, end;
    float time;
                                      
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start, 0);
    
    /* First need to load index into memory */
    read_index("index.txt");

    unsigned long list[3] = {17756674167933085944UL, 17756674167933124018UL,  17756674167933154478};
    // read all files given a list of ids
    if (list != NULL)
    {
        load_data_by_ids(list, 3, &db);  
    }
    else
    {
        perror("read files error!");
        return EXIT_FAILURE;
    }

    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);

    cudaEventElapsedTime(&time, start, end);

    // get end points
    thrust::device_vector<double> out_lon_vec(db.size);
    thrust::device_vector<double> out_lat_vec(db.size);
    collect_end_points(db.col_lon_vec,
                       db.col_lat_vec,
                       db.col_state_vec,
                       out_lon_vec,
                       out_lat_vec
    );

    // remove useless items
    thrust::device_vector<double>::iterator new_end = 
                thrust::remove_if(out_lon_vec.begin(), out_lon_vec.end(), thrust::placeholders::_1 == 0);
    out_lon_vec.erase(new_end, out_lon_vec.end());

    new_end = thrust::remove_if(out_lat_vec.begin(), out_lat_vec.end(), thrust::placeholders::_1 == 0);
    out_lat_vec.erase(new_end, out_lat_vec.end());

    //thrust::copy(out_lon_vec.begin(), out_lon_vec.end(), std::ostream_iterator<double>(std::cout, ","));
    //thrust::copy(out_lat_vec.begin(), out_lat_vec.end(), std::ostream_iterator<double>(std::cout, ","));

    // copy points from GPU
    thrust::host_vector<double> h_lon_vec(out_lon_vec.size());
    thrust::copy(out_lon_vec.begin(), out_lon_vec.end(), h_lon_vec.begin());

    thrust::host_vector<double> h_lat_vec(out_lat_vec.size());
    thrust::copy(out_lat_vec.begin(), out_lat_vec.end(), h_lat_vec.begin());

    struct point_t * points = (struct point_t *) malloc(h_lon_vec.size() * sizeof(struct point_t));
    
    for (int i = 0; i < h_lon_vec.size(); i++) 
    {
        points[i].lon = h_lon_vec[i];
        points[i].lat = h_lat_vec[i];
        points[i].cluster_id = UNCLASSIFIED;
    }  

    // dbscan to find areas
    //dbscan(points, h_lon_vec.size(), 0.01, 5, euclidean_dist);
   
    //print_points(points, h_lat_vec.size());

    //struct point_t * center_points = sum_cluster(points, h_lon_vec.size());

    free(points);
    //free(center_points);

    //cudaEventCreate(&start);
    //cudaEventCreate(&end);
                
    //cudaEventRecord(start, 0);
    
    thrust::device_vector<int> tid_vec = construct_quadtree("/home/wangfei/201401TRK/TRK20140101/CC2");

    //cudaEventRecord(end, 0);
    //cudaEventSynchronize(end);
                
    //cudaEventElapsedTime(&time, start, end);               
    //cout << "build index elapsing " << time << " milli-seconds \n";

    float rect[4] = {120.459, 27.849, 120.8, 27.9};

    //cudaEventCreate(&start);
    //cudaEventCreate(&end);
                
    //cudaEventRecord(start, 0);
    
    thrust::host_vector<int> result_vec( tid_vec.size() );
    search( rect, 1000l, 2000l, thrust::raw_pointer_cast( &result_vec[0]), tid_vec );
    
    //cudaEventRecord( end, 0 );
    //cudaEventSynchronize( end );
    //cudaEventElapsedTime( &time, start, end );
    //cout << "search trajectory elapsing " << time << " milli-seconds \n";
 
    return 0;
}
