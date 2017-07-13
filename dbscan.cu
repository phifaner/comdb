#include <thrust/copy.h>

#include "dbscan.h"

int collect_end_points(thrust::device_vector<double> in_lon_vec, 
            thrust::device_vector<double> in_lat_vec,
            thrust::device_vector<int> in_state_vec,
            thrust::device_vector<double> &out_lon_vec, 
            thrust::device_vector<double> &out_lat_vec)
{
    /* handover to GPU */
    //double * in_lon_data = thrust::raw_pointer_cast(in_lon_vec.data());
    //double * in_lat_data = thrust::raw_pointer_cast(in_lat_vec.data());
    //double * out_lon_data = thrust::raw_pointer_cast(out_lon_vec.data());
    //double * out_lat_data = thrust::raw_pointer_cast(out_lat_vec.data());
    int * in_state_data = thrust::raw_pointer_cast(in_state_vec.data());
    
    /* record the positions of state change */
    thrust::device_vector<int> index_vec(in_lon_vec.size());
    int * index_data = thrust::raw_pointer_cast(index_vec.data());

    end_points endpts(in_state_data, index_data);

    thrust::counting_iterator<unsigned int> begin(0);
    thrust::counting_iterator<unsigned int> end(in_lon_vec.size() - 1);

    thrust::for_each(begin, end, endpts);

    /* handover end points from in to out vector */
    thrust::copy_if(
            in_lon_vec.begin(), 
            in_lon_vec.end(), 
            index_vec.begin(),   /* stencil */ 
            out_lon_vec.begin(), 
            thrust::placeholders::_1 == 1
    );

    thrust::copy_if(
            in_lat_vec.begin(), 
            in_lat_vec.end(), 
            index_vec.begin(),   /* stencil */ 
            out_lat_vec.begin(), 
            thrust::placeholders::_1 == 1
    );

    //thrust::copy(out_lon_vec.begin(), out_lon_vec.end(), std::ostream_iterator<double>(std::cout, ","));

    return 1;
}

struct point_t * sum_cluster(struct point_t * points, unsigned int num_points)
{
    unsigned int num_cluster = 0;
    unsigned int bitmap[50] = { 0 };        // to sum number of clusters

    // scan points to set bitmap according to cluster id
    for (int i = 0; i < num_points; i++)
    {
        int cluster_id = points[i].cluster_id;

        if (cluster_id < 1) continue;
        
        if (bitmap[cluster_id] == 0) bitmap[cluster_id] = 1;
            
    }

    // sum cluster
    for (int i = 0; i < 50; i++)
    {
        if (bitmap[i] == 1) ++num_cluster;    
    }

    struct point_t * center_points = 
                (struct point_t *) calloc(num_cluster, sizeof(struct point_t));

    // calculate center of cluster and sum of points
    unsigned int k = 0;
    for (int i = 0; i < num_points; i++)
    {
        int cluster_id = points[i].cluster_id;
        
        if (cluster_id < 1) continue;

        // set initialized values
        center_points[k].lon = points[i].lon;
        center_points[k].lat = points[i].lat;
        center_points[k].cluster_id = 1;        // used to count

        for (int j = 0; j < num_points; j++)
        {
            // i and j belong to the same cluster
            if (i != j && points[j].cluster_id == cluster_id)
            {
                center_points[k].lon += points[j].lon;
                center_points[k].lat += points[j].lat;
                center_points[k].cluster_id += 1;

                // set the piont as visited
                points[j].cluster_id = -3;
            }
        }

        k++;
            
    }

    // compute center point's coordinates
    for (int i = 0; i < num_cluster; i++)
    {
        center_points[i].lon /= center_points[i].cluster_id;
        center_points[i].lat /= center_points[i].cluster_id;

        printf("center point %d: lon: %lf, lat: %lf, num: %d\n", 
                i, center_points[i].lon, center_points[i].lat, center_points[i].cluster_id);
    }

    return center_points;    
}

/* TODO change to CUDA code */
int dbscan(struct point_t *points, 
        unsigned int num_points,
        double epsilon,
        unsigned int minpts,
        double (*dist) (struct point_t * a, struct point_t * b))
{
    /* */
    unsigned int i, cluster_id = 0;
 
    for (i = 0; i < num_points; ++i)
    {
        if (points[i].cluster_id == UNCLASSIFIED)
        {
            if (expand(i, cluster_id, points,
                       num_points, epsilon, minpts,
                       dist) == CORE_POINT)
            {
                ++cluster_id;
                printf("dbscan cluster id -------%d\n", cluster_id);
     
            }
        }
    }

    return 1;
}

int expand(
        unsigned int index,
        unsigned int cluster_id,
        struct point_t * points,
        unsigned int num_points,
        double epsilon,
        unsigned int minpts,
        double (*dist) (struct point_t * a, struct point_t * b))
{
    int return_value = NOT_CORE_POINT;
    epsilon_neighbors_t * seeds = 
        get_epsilon_neighbors(index, points, num_points, epsilon, dist);

    if (seeds == NULL) return FAILURE;

    if (seeds->num_members < minpts)
        points[index].cluster_id = NOISE;
    else
    {
        points[index].cluster_id = cluster_id;
        node_t * h = seeds->head;
        
        /* set cluster id of cache points in the same cluster */
        while (h)
        {
            points[h->index].cluster_id = cluster_id;
            h = h->next;

        }
        
        /* spread cache points */
        h = seeds->head;
        while (h)
        {
            spread(h->index, seeds, cluster_id,
                    points, num_points, epsilon, minpts, dist);
            h = h->next;
        }
        
        return_value = CORE_POINT;   
    }

    destroy_epsilon_neighbors(seeds);

    return return_value;
}

int spread(unsigned int index,
           epsilon_neighbors_t * seeds,
           unsigned int cluster_id,
           struct point_t * points,
           unsigned int num_points,
           double epsilon,
           unsigned int minpts,
           double  (*dist) (struct point_t * a, struct point_t * b))
{
    epsilon_neighbors_t * spread = 
            get_epsilon_neighbors(index, points, num_points, epsilon, dist);

    if (spread == NULL) return FAILURE;

    /**/
    if (spread->num_members >= minpts)
    {
        node_t * n = spread->head;
        struct point_t * d;
        
        while (n)
        {
            d = &points[n->index];
            if (d->cluster_id == NOISE || 
                d->cluster_id == UNCLASSIFIED)
            {
                if (d->cluster_id == UNCLASSIFIED)
                {
                    if (append_at_end(n->index, seeds) == FAILURE)
                    {
                        destroy_epsilon_neighbors(spread);
                        return FAILURE;
                    }
                }

                d->cluster_id = cluster_id;
                //printf("function spread cluster id %d\n", cluster_id);
            }

           n = n->next; 
        }
    }

    destroy_epsilon_neighbors(spread);
    return SUCCESS;
}

double euclidean_dist(struct point_t * a, struct point_t * b)
{
    return sqrt(pow(a->lon - b->lon, 2) + pow(a->lat - b->lat, 2) /*+ pow(a->z - b->z, 2)*/);
}

node_t * create_node(unsigned int index)
{
    node_t * n = (node_t *) calloc(1, sizeof(node_t));
    if (n == NULL)
        perror("Failed to allocate node.");
    else
    {
        n->index = index;
        n->next = NULL;
    }

    return n;
}

int append_at_end(unsigned int index, epsilon_neighbors_t * en)
{
    node_t * n = create_node(index);
    if (n == NULL)
    {
        free(en);
        return FAILURE;
    }

    if (en->head == NULL)
    {
        en->head = n;
        en->tail = n;
    }
    else
    {
        en->tail->next = n;
        en->tail = n;
    }

    ++(en->num_members);
    return SUCCESS;
}

epsilon_neighbors_t * get_epsilon_neighbors(
            unsigned int index,
            struct point_t *points,
            unsigned int num_points,
            double epsilon,
            double (*dist)(struct point_t * a, struct point_t * b))
{
    epsilon_neighbors_t * en =
            (epsilon_neighbors_t *) calloc(1, sizeof(epsilon_neighbors_t));
    if (en == NULL) {
        perror("Failed to allocate epsilon neighbours.");
        return en;
    }

    /* compute each points and find neighbors */
    for (int i = 0; i < num_points; ++i) 
    {
        if (i == index) continue;
        if (dist(&points[index], &points[i]) > epsilon) continue;
        else 
        {
            if (append_at_end(i, en) == FAILURE) 
            {            
                destroy_epsilon_neighbors(en);             
                en = NULL;                                               
                break;
            }
        }
    }       
 
    return en;
}   

void destroy_epsilon_neighbors(epsilon_neighbors_t *en)
{
    if (en) 
    {
        node_t *t, *h = en->head;
        
        while (h) 
        {
            t = h->next;
            free(h);
            h = t;
        }

        free(en);
    }
}

/*unsigned int parse_input(
            FILE *file,
            struct point_t **points,
            double *epsilon,
            unsigned int *minpts)
{
    unsigned int num_points, i = 0;
   
    fscanf(file, "%lf %u %u\n",
            epsilon, minpts, &num_points);

    point_t *p = (point_t *)
        calloc(num_points, sizeof(struct point_t));

    if (p == NULL) {
        perror("Failed to allocate points array");
        return 0;
    }

    while (i < num_points) {
        fscanf(file, "%lf %lf\n",
                &(p[i].lon), &(p[i].lat), &(p[i].z));

        p[i].cluster_id = UNCLASSIFIED;
        ++i;
        
    }

    *points = p;

    return num_points;
}*/

void print_points(
        struct point_t *points,
        unsigned int num_points)
{

    unsigned int i = 0;

    printf("Number of points: %u\n"
            " x     y      cluster_id\n"
            "------------------------\n"
            , num_points);

    while (i < num_points) {
        printf("%5.4lf %5.4lf: %d\n",
                points[i].lon,
                points[i].lat, 
               // points[i].z,
                points[i].cluster_id);
        ++i;
    }
}

/*int main(void) {
   struct point_t *points;
   double epsilon;
   unsigned int minpts;

   unsigned int num_points =
       parse_input(stdin, &points, &epsilon, &minpts);

   if (num_points) {

       dbscan(points, num_points, epsilon,
               minpts, euclidean_dist);

       printf("Epsilon: %lf\n", epsilon);
       printf("Minimum points: %u\n", minpts);
       print_points(points, num_points);

   }

   free(points);

   return 0;
}*/
