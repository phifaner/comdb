#include <math.h>
#include <thrust/device_vector.h>

#define NOISE           -2
#define CORE_POINT       1


#define UNCLASSIFIED    -1
#define NOT_CORE_POINT   0

#define SUCCESS          0
#define FAILURE         -3

struct point_t
{
    double lon, lat;
    int cluster_id;
};

typedef struct node_s 
{
    unsigned int index;
    struct node_s * next;
} node_t;


typedef struct epsilon_neighbors 
{
    unsigned int num_members;
    node_t * head, * tail;
} epsilon_neighbors_t;


/* collect end points of a trajectory w.r.t the state attribute */
int collect_end_points(thrust::device_vector<double> in_lon_vec,
            thrust::device_vector<double> in_lat_vec,
            thrust::device_vector<int> in_state_vec,
            thrust::device_vector<double> &out_lon_vec, 
            thrust::device_vector<double> &out_lat_vec);


/* compute number of points of each cluster 
 * return cluster's center and its sum    */
struct point_t * sum_cluster(struct point_t *, unsigned int);

int dbscan(struct point_t *, 
        unsigned int num_points,
        double epsilon,
        unsigned int minpts,
        double (*dist) (struct point_t *, struct point_t *));

int expand(
        unsigned int index,
        unsigned int cluster_id,
        struct point_t * points,
        unsigned int num_points,
        double epsilon,
        unsigned int minpts,
        double (*dist) (struct point_t *, struct point_t *));

double euclidean_dist(struct point_t *, struct point_t *);

double adjacent_intensity_dist(struct point *, struct point_t *);

int spread(unsigned int index,
        epsilon_neighbors_t * seeds,
        unsigned int cluster_id,
        struct point_t * points,
        unsigned int num_points,
        double epsilon,
        unsigned int minpts,
        double (*dist)(struct point_t *, struct point_t *));

int append_at_end(unsigned int index, epsilon_neighbors_t *);

epsilon_neighbors_t * get_epsilon_neighbors(
        unsigned int index,         /* point index in point set */
        struct point_t * points,    /* the point set */
        unsigned int num_points,
        double epsilon,
        double (*dist) (struct point_t *, struct point_t *));

void destroy_epsilon_neighbors(epsilon_neighbors_t *);

void print_points(struct point_t * points, unsigned int num_points);


/* find end points of a trajectory */
struct end_points
{
    int * state;
    int * index;

    end_points(int * s, int * i) : state(s), index(i) {}

    __host__ __device__
    void operator() (const int i)
    {    
        if (state[i] != state[i+1])
        {
            /* from available to hired */
            if (state[i] == 0) index[i+1] = 1;
            else if (state[i] == 1) index[i] = 1;
 
        }        
    }
};
