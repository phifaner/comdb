#ifndef COM_R_TREE_H
#define COM_R_TREE_H

#include <thrust/device_vector.h>

struct rtree_point
{
    unsigned long   tid;        /* trajectory id */
    char    *       sstime;     /* starting date */
    char    *       eetime;     /* end date */
};

struct rtree_rect
{
    double  lon_low, lat_low;
    double  lon_upper, lat_upper;
};


struct rtree_node
{
    rtree_rect      bbox;
    size_t          num;
    rtree_node *    children;
};

struct rtree_leaf
{
    rtree_rect      bbox;
    size_t          num;
    rtree_point *   points;
};

struct bbox
{
    double x_min, y_min, x_max,  y_max;
};

void init_file_array();
void list_dir(const char *, int);

/* compute each subtrajectory's bounding box in a path
 * return number of bbox */
void traj_bbx(thrust::device_vector<float4> &);

/* initialize index array and rect array 
 * to store R-Tree items and bbox
 * item_num: subtrajectory number in the path */
void init(int fanout);

/* construct R Tree in a bulk loading method, including sorting and packing */
void sort_lowx(thrust::device_vector<float4> &rect_vec);

void pack_level(
        thrust::device_vector<float4>, 
        int, int, 
        thrust::device_vector<int> &,
        thrust::device_vector<float4> &
        );

int compute_bbx(char *, float4 &);

void compute_index(int K, int B);

void bulk_load(int fanout, thrust::device_vector<float4> & bbx);
void search();

/*------------------------CUDA Kernel-------------------------------*/
struct pack_kernel
{
    
    /* input node item bbox */
    float4 * bbox;
    //lon_min_vec, lat_min_vec;
    //double * lon_max_vec, lat_max_vec;

    /* fanout of tree node */
    int fanout;
    int * index_vec;

    /* output parent bbox */
    float4 * out_bbox;
    //double * out_lon_min_vec, out_lat_min_vec;
    //double * out_lon_max_vec, out_lat_max_vec;

    pack_kernel(
            float4 * _bbx,
            //double * lon_min,
            //double * lat_min,
            //double * lon_max,
            //double * lat_max,
            int fan,
            int * index)
            //float4 * _out_bbx)
            //double * out_lon_min,
            //double * out_lat_min,
            //double * out_lon_max,
            //double * out_lat_max)
        :
         bbox(_bbx),
         //lon_min_vec(lon_min),
         //lat_min_vec(lat_min),
         //lon_max_vec(lon_max),
         //lat_max_vec(lat_max),
         fanout(fan),
         index_vec(index) {}
         //out_bbox(_out_bbx) {}
         //out_lon_min_vec(out_lon_min),
         //out_lat_min_vec(out_lat_min),
         //out_lon_max_vec(out_lon_max),
         //out_lat_max_vec(out_lat_max) {}

    __host__ __device__
    void operator() (const int & i)
    {
        // i = k * fanout
        if (!i%fanout) return;

        
        // deal with items in a node
        #pragma unroll
        for (int j = 0; j < fanout; j++)
        {
            // parent index array
            index_vec[i+j] = i + j;

            printf("-------%d\n", i+j);

            // parent bounding box
            //float4 bbox = bounding_box(i);
            //out_lon_min_vec[i] = bbox[0];
            //out_lat_min_vec[i] = bbox[1];
            //out_lon_max_vec[i] = bbox[2];
            //out_lat_max_vec[i] = bbox[3];

        }

    }

    __host__ __device__
    float4 bounding_box(int i)
    {
        double lon_min, lat_min, lon_max, lat_max;

        for (int j = 0; j < fanout; j++)
        {
            if (bbox[i+j].x < lon_min) lon_min = bbox[i+j].x;
            if (bbox[i+j].y < lat_min) lat_min = bbox[i+j].y;
            if (bbox[i+j].z > lon_max) lon_max = bbox[i+j].z;
            //if (lon_max_vec[i+j] > lon_max) lon_max = lon_max_vec[i+j];
            if (bbox[i+j].w > lat_max) lat_max = bbox[i+j].w;
        }

        return make_float4(lon_min, lat_min, lon_max, lat_max);
    }

    /*__host__ __device__
    int ceiling_log(int k, int b)
    {
        int m = log(k)/log(b);

        if (log(k)/log(b) <= m) return m+1;
        else return m;
    }*/
};



struct float4_greater
{
    __host__ __device__
    bool operator() (const float4 &lhs, const float4 &rhs) const {return lhs.x > rhs.x;}
};
#endif
