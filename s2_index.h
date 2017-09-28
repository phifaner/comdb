#include <thrust/device_vector.h>

#include "s2/base/integral_types.h"
#include "s2/s2cellid.h"
#include "rtree.h"
#include "poi.h"
#include "mix.h"

struct Parameters
{
    // number of cells of a trajectory
    const size_t num_cells;
    // number of poi cells.
    const size_t num_poi_cells;    
    // number of poi types
    const size_t num_poi_types;


    // Constructor set to default values.
    __host__ __device__ Parameters( size_t _num_cells, size_t _num_poi_cells, size_t _num_poi_types ) : 
        num_cells(_num_cells), 
        num_poi_cells(_num_poi_cells),
        num_poi_types(_num_poi_types)
    {}
};


class Inv_list
{
    protected:
        // associated cell id
        long cell_id;
        
        // time interval
        long start, end;
        
        // // trajectory id list
        int *id_array;
        
        // // size of id list
        int num;
       
       
    public:
     
        // get id list
        int *get_id_array(long cid) { return id_array; }
        
};
        
// index of all       
class Movix       
{     
    protected:
        // spatial region of a city map
        float min_lon, min_lat, max_lon, max_lat;
        
        // temporal span of all data set
        long start, end;
        
        
        // inverted list of the index
        Inv_list *p_inv_list;
        
        // Trajectory on GPU
        thrust::device_vector<int>              tid_vec;
        thrust::device_vector<uint64>           cell_vec;
        thrust::device_vector<unsigned long>    ts_vec;
        
        
    public:
   
        // initialize to set data
        void set_map(float _min_lon, float _min_lat, float _max_lon, float _max_lat)
        {
            min_lon = _min_lon;
            min_lat = _min_lat;
            max_lon = _max_lon;
            max_lat = _max_lat;
        }
       
        void set_time_span(long s, long e)
        {
            start = s;
            end = e;
        }
        
        // build index using RTree
        RTree build_index_rtree(const char *path);
        
        // build bitmap index based on s2's level, path's name including the level
        HMix build_bitmap_index(const char *path);

        void cuda_build_bitmap_index(const char *path);
        
        // cost estimation
        // 1. given a time interval and cells, estimate the number of result
        // 2. given a collection of cells and their covering areas 
        std::map<int, H_Index> cost_by_time(HMix *H, std::vector<std::vector<uint64> > V, unsigned long *T, double v);
        std::map<int, std::vector<unsigned long> >  cost_by_types(HMix *H, std::vector<std::vector<uint64> > type_cells, std::vector<unsigned long> times, std::vector<float> radius);

        // return trajectory ids, which are on device
        thrust::device_vector<int> cuda_cost_by_types(std::vector<std::vector<uint64> > type_cells, std::vector<unsigned long> times);
        
        // search index
        // given several cell id and a time span
        // return a list of trajectory id
        std::vector<int> search_RTree(RTree *tree, std::vector<uint64> cid_vec, long s, long e);

        std::vector<int> search(HMix *H, std::vector<uint64> cells, unsigned long *T);

        // given name of POI and a time span
        // return a list of trajectory id
        std::vector<int> poi_search_RTree(RTree *tree, POI_Data *D, std::vector<S2_POI> P, long s, long e);

        std::vector<int> poi_search(HMix *H, HMix *LH, POI_Data *D, std::vector<S2_POI> P, std::vector<unsigned long> T, double v);
        //get_extent(uint64 cell_id, )

    //private:
        int read_cell_points(char *);

        double measure(double lat1, double lon1, double lat2, double lon2);

        size_t length; 

        int level;
};


inline uint32_t highBytes(const uint64_t in) { return uint32_t(in >> 32); }
