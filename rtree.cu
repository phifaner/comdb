#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>

#include <algorithm>

#include "rtree.h"

__host__ __device__
int overlap(RTree_Rect *R, RTree_Rect * S)
{
    register RTree_Rect *r = R, *s = S;

    assert(r && s);
    

    if ( r->left > s->right || r->right < s->left
           || r->top > s->bottom || r->bottom < s->top )
    {

    //printf("overlap R: %llu, %llu, %lu, %lu, S: %llu, %llu, %lu, %lu\n", 
    //        r->left, r->right, r->top, r->bottom, s->left, s->right, s->top, s->bottom);
        return 0;
    }
    else 
        return 1;
}

__host__ __device__
int contains(RTree_Rect *R, RTree_Point *P)
{
    register RTree_Rect *r = R;
    register RTree_Point *p = P;

    assert(r && p);

    //printf("point: %llu, %lu, Rect: %llu, %llu, %lu, %lu\n", 
    //        p->x, p->y, r->left, r->right, r->top, r->bottom);

    if (p->x < r->right && p->x > r->left 
            && p->y < r->bottom && p->y > r->top)
        return 1;
    else 
        return 0;
}

__host__ __device__
inline void init_boundary(RTree_Rect *bbox)
{
    bbox->top = ULONG_MAX;
    bbox->bottom = -ULONG_MAX;
    bbox->left = ULLONG_MAX;
    bbox->right = -ULLONG_MAX;
}

__host__ __device__
inline void update_boundary(RTree_Rect *bbox, RTree_Rect *node_bbx)
{
    bbox->top = min(bbox->top, node_bbx->top);
    bbox->bottom = max(bbox->bottom, node_bbx->bottom);
    bbox->left = min(bbox->left, node_bbx->left);
    bbox->right = max(bbox->right, node_bbx->right);

    //printf("---node bbox: %llu, %llu, update: %llu, %llu\n", 
    //        node_bbx->left, node_bbx->right, bbox->left, bbox->right);

}

__host__ __device__
inline void c_update_boundary(RTree_Rect *bbox, RTree_Point *p)
{
    bbox->top = min(p->y, bbox->top);
    bbox->bottom = max(p->y, bbox->bottom);
    bbox->left = min(p->x, bbox->left);
    bbox->right = max(p->x, bbox->right);

    //printf("x: %llu, bbox: %lu, %lu, %llu, %llu\n", p->x, bbox->top, bbox->bottom, bbox->left, bbox->right);
}

__host__ __device__
inline size_t get_node_length (
        const size_t i, 
        const size_t len_level,
        const size_t previous_level_len,
        const size_t node_size)
{
    const size_t n = node_size;
    const size_t len = previous_level_len;
    const size_t final_i = len_level -1;

    // set lnum to len%n if it's the last iteration and there's a remainder, else n
    return ((i != final_i || len % n == 0) *n) + ((i == final_i && len % n != 0) * (len % n));
}

// points are on device and sorted by x
void cuda_sort(RTree_Points *sorted)
{ 
    uint64 *X = sorted->X;
    unsigned long *Y = sorted->Y;
    int *ID = sorted->ID;

    // sort by x
    auto tbegin = thrust::make_zip_iterator(thrust::make_tuple(Y, ID));
    auto tend = thrust::make_zip_iterator(thrust::make_tuple(Y+sorted->length, ID+sorted->length));
    thrust::sort_by_key(thrust::device, X, X+sorted->length, tbegin);

}


RTree cuda_create_rtree(RTree_Points points)
{
    cuda_sort(&points);
    RTree_Leaf *leaves = cuda_create_leaves( &points );
    const size_t len_leaf = DIV_CEIL(points.length, RTREE_NODE_SIZE);

    // build rtree from bottom
    RTree_Node *level_previous  = (RTree_Node*) leaves;
    size_t      len_previous    = len_leaf;
    size_t      depth           = 1;    // leaf level: 0
    size_t      num_nodes       = len_leaf;
    while (len_previous > RTREE_NODE_SIZE)
    {
        level_previous = cuda_create_level(level_previous, len_previous, depth);
        num_nodes += level_previous->num;
        len_previous = DIV_CEIL(len_previous, RTREE_NODE_SIZE);
        ++depth;
    }

    // tackle the root node
    RTree_Node *root = new RTree_Node();
    init_boundary(&root->bbox);
    root->num = len_previous;
    root->children = level_previous;
    num_nodes += root->num;
    for (size_t i = 0, end = len_previous; i != end; ++i)
        update_boundary(&root->bbox, &root->children[i].bbox);
    ++depth;
    root->depth = depth;

    RTree tree = {depth, root};
    return tree;
}

__global__
void create_level_kernel
        (
            RTree_Node *next_level,
            RTree_Node *nodes,
            RTree_Node *real_nodes,
            const size_t len,
            size_t depth
         )
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t next_level_len = DIV_CEIL(len, RTREE_NODE_SIZE);
 
    if (i >= next_level_len) return;    // skip the final block remainder

    RTree_Node *n = &next_level[i];
    init_boundary(&n->bbox);
    n->num = get_node_length(i, next_level_len, len, RTREE_NODE_SIZE);
    n->children = &real_nodes[i * RTREE_NODE_SIZE]; 
    n->depth = depth;   
    //printf("level num: %d, ---num: %lu\n", n->num, next_level_len);

  #pragma unroll
    for (size_t j = 0, jend = n->num; j != jend; ++j)
    {
        update_boundary(&n->bbox, &nodes[i * RTREE_NODE_SIZE + j].bbox);
        //printf("after set node bbox: %lu, %lu, %llu, %llu\n", 
        //    n->bbox.top, n->bbox.bottom, n->bbox.left, n->bbox.right);
    }
}

RTree_Node* cuda_create_level(RTree_Node *nodes, const size_t len, size_t depth)
{
    const size_t THREADS_PER_BLOCK = 512; 
    const size_t next_level_len = DIV_CEIL(len, RTREE_NODE_SIZE);

    RTree_Node *d_nodes;
    RTree_Node *d_next_level;
    cudaMalloc( (void**) &d_nodes, len * sizeof(RTree_Node) );
    cudaMalloc( (void**) &d_next_level, next_level_len * sizeof(RTree_Node) );

    cudaMemcpy(d_nodes, nodes, len * sizeof(RTree_Node), cudaMemcpyHostToDevice);
    
    create_level_kernel<<< (next_level_len + (THREADS_PER_BLOCK - 1)) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>
            (d_next_level, d_nodes, nodes, len, depth);    
   
    RTree_Node *next_level = new RTree_Node[next_level_len];
    cudaMemcpy(next_level, d_next_level, next_level_len * sizeof(RTree_Node), cudaMemcpyDeviceToHost);

    cudaFree(d_next_level);
    cudaFree(d_nodes);

    return next_level;
}

__global__
void create_leaves_kernel
        (
            RTree_Leaf      *leaves,
            RTree_Point     *points,
            RTree_Point     *h_points,
            uint64          *X,
            unsigned long   *Y,
            int             *ID,
            const size_t    len
        )
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    const size_t len_leaf = DIV_CEIL(len, RTREE_NODE_SIZE);
    if (i >= len_leaf) return;  // skip the final block remainder

    // tackle leaf points
    RTree_Leaf *l = &leaves[i];
    init_boundary(&l->bbox);
    l->num = get_node_length(i, len_leaf, len, RTREE_NODE_SIZE);
    l->depth = 0;
    l->points = &h_points[i * RTREE_NODE_SIZE]; // occupy position

    // compute MBR from its 
  #pragma unroll
    for (size_t j = 0, jend = l->num; j != jend; ++j)
    {
        // *** use pointer, not value ***/
        RTree_Point *p = &points[i* RTREE_NODE_SIZE + j];
        p->x     = X[i   * RTREE_NODE_SIZE + j];
        p->y     = Y[i   * RTREE_NODE_SIZE + j];
        p->id    = ID[i  * RTREE_NODE_SIZE + j];

        //printf("----------id: %d, j: %lu\n", p->id, j);
        c_update_boundary(&l->bbox, p);
    }
}

RTree_Leaf* cuda_create_leaves(RTree_Points *sorted)
{
    const size_t THREADS_PER_BLOCK = 512;
    
    const size_t len = sorted->length;
    const size_t num_leaf = DIV_CEIL(len, RTREE_NODE_SIZE);

    RTree_Leaf  *d_leaves;
    RTree_Point *d_points;

    cudaMalloc( (void**) &d_leaves, num_leaf    * sizeof(RTree_Leaf) );
    cudaMalloc( (void**) &d_points, len         * sizeof(RTree_Point) );

    // points on host will be passed to kernel and only occupy the position
    RTree_Point *points = new RTree_Point[len];

    create_leaves_kernel<<< (num_leaf + (THREADS_PER_BLOCK - 1)) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >>>
        (d_leaves, d_points, points, sorted->X, sorted->Y, sorted->ID, len);

    RTree_Leaf  *leaves = new RTree_Leaf[num_leaf];

    // copy points from device to host
    cudaMemcpy(leaves, d_leaves, num_leaf   * sizeof(RTree_Leaf), cudaMemcpyDeviceToHost);
    cudaMemcpy(points, d_points, len        * sizeof(RTree_Point), cudaMemcpyDeviceToHost);

    cudaFree(d_leaves);
    cudaFree(d_points);

    return leaves;

}

int cpu_search(RTree_Node *N, RTree_Rect *rect, std::vector<int> &points)
{
    register RTree_Node *n = N;
    register RTree_Rect *r = rect;
    register int hit_count = 0;
    register int i;

    assert(n);
    assert(n->num);
    assert(r);
    
    //printf("depth: %lu, bbox: %llu, %llu, %lu, %lu\t rect: %llu, %lu\n", n->depth, n->bbox.left, n->bbox.right,
    //     n->bbox.top, n->bbox.bottom, r->left, r->top );


    if (n->depth > 0)
    {
        for (i = 0; i < n->num; i++)
        {
            printf("depth: %lu, bbox: %llu, %llu, %lu, %lu\t rect: %llu, %lu\t num: %lu\n", 
                    n->depth, n->children[i].bbox.left, n->children[i].bbox.right,
                        n->children[i].bbox.top, n->children[i].bbox.bottom, r->left, r->top
                        , n->children[i].num );

            if ( overlap(r, &n->children[i].bbox) )
            {
                hit_count += cpu_search(&n->children[i], rect, points);
            }
        }
    }
    else    // this is a leaf node
    {
        if ( n->num && overlap(r, &n->bbox) )
        {

            //printf("---%llu, %llu, %lu, %lu\n", n->bbox.left, n->bbox.right, n->bbox.top, n->bbox.bottom);
            
            RTree_Leaf *l = (RTree_Leaf*) n;
            for (i = 0; i < n->num; i++)
            {
                // determine whether points in rect
                if ( contains(r, &l->points[i] ) )
                {
                    hit_count++;
                    
                    // check if contains this point
                    if ( std::find(points.begin(), points.end(), l->points[i].id) == points.end() )
                        points.push_back(l->points[i].id);
                  
                    printf("%d trajectory is hit, %llu, %lu\n", l->points[i].id, l->points[i].x, l->points[i].y);
                }
            }
        }
    }

    return hit_count;
}

template< int MAX_THREADS_PER_BLOCK >
__global__ 
void search_kernel(
        CUDA_RTree_Node     *   d_nodes,
        int                 *   d_edges,
        RTree_Rect          *   d_rects,
        bool                *   d_search_front,
        RTree_Rect          *   rects, 
        int                 *   results, 
        int                     num_nodes)
{
    // shared memory to store the query rectangles
    extern __shared__ RTree_Rect rmem[];
    
    // Address of shared memory
    RTree_Rect *s_rect = (RTree_Rect *) &rmem[blockIdx.x];

    // each thread represents one node
    int tid = blockIdx.x * MAX_THREADS_PER_BLOCK + threadIdx.x;

    // whether the query rectangle overlaps the MBR of the frontier node
    bool flag = false;   
    if ( overlap(&d_rects[tid], s_rect) ) flag = true;

    // node is in frontier and its MBR overlaps query rectangle
    if (tid < num_nodes && d_search_front[tid] && flag)
    {
        // remove it from frontier
        d_search_front[tid] = false;

        // reach Leaf level
        if (d_nodes[tid].starting == -1)
        {
            results[tid] = 1;
            return ;
        }

        // put its children to the next search_front
        for (int i = d_nodes[tid].starting; i < (d_nodes[tid].num_edges + d_nodes[tid].starting); i++)
        {
            int id = d_edges[i];
            d_search_front[id] = true;
        }
    }

    search_kernel<MAX_THREADS_PER_BLOCK><<<10, 20>>>
        (d_nodes, d_edges, d_rects, d_search_front, rects,  results, num_nodes);

}

void fill_edges(RTree_Node *N, CUDA_RTree_Node *h_nodes, int *h_edges, RTree_Rect *h_rects, int& node_id)
{
    register RTree_Node * n = N;

    if (node_id == 0)
    {
        h_nodes[node_id].starting = 0;  // initialize root node
        
        for (int i = h_nodes[0].starting; i < (h_nodes[0].starting + n->num); i++)
        {

            // starting index of child in array 
            if (i == 0) 
                h_edges[i] = RTREE_NODE_SIZE;
            else
                h_edges[i] = n->num;
        
        }
    }
    else
    {
        
        if (n->depth > 0) // set nodes
        {
            h_nodes[node_id].starting = h_nodes[node_id-1].starting + h_nodes[node_id-1].num_edges;

            for (int i = h_nodes[node_id].starting; i < (h_nodes[node_id].starting + n->num); i++)
            {
                // starting index of child in array 
                h_edges[i] = h_edges[i-1] + h_nodes[node_id-1].num_edges;
        
            }
        }
        else    // set Leaf node
        {
            h_nodes[node_id].starting = -1;   
        }
    }

    h_nodes[node_id].num_edges = n->num;
    h_rects[node_id] = n->bbox;

    // recursively fill edges
    for (int i = 0; i < n->num; i++)
    {
        fill_edges(&n->children[i], h_nodes, h_edges, h_rects, ++node_id);   
    }

}

RTree_Points cuda_search(RTree *tree, std::vector<RTree_Rect> rect_vec)
{
    CUDA_RTree_Node *   h_nodes = (CUDA_RTree_Node *) malloc(tree->num * sizeof(CUDA_RTree_Node));
    int *               h_edges = (int *) malloc(tree->num * sizeof(int) * RTREE_NODE_SIZE);
    RTree_Rect      *   h_rects = (RTree_Rect *) malloc(tree->num * sizeof(RTree_Rect));

    int node_id = 0;


    printf("tree node number: %lu-----\n", tree->num);

    // copy data from cpu to gpu
    fill_edges(tree->root, h_nodes, h_edges, h_rects, node_id);   
    
    for (int i = 0; i < tree->num; i++)
    {
        printf("starting of node: %d is %d\n", i, h_nodes[i].starting);
    }
    // allocate n blocks to deal with n query rectangles

    RTree_Points points;

    return points;
}

