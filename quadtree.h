#ifndef     __QUADTREE_H
#define      __QUADTREE_H

#include <thrust/device_vector.h>
////////////////////////////////////////////////////////////////////////////////
//// A structure of trajectory leaf nodes (structure of arrays).
//////////////////////////////////////////////////////////////////////////////////
class Leaf_nodes
{
    float   *m_min_x, *m_max_x;     // bounding box of x
    float   *m_min_y, *m_max_y;     // bounding box of y
    long    *m_start_t, *m_end_t;   // starting time and end time
    int     *m_tid;                 // trajectory id

public:
    // Constructor
    __host__ __device__ Leaf_nodes() 
        : m_min_x(NULL), m_max_x(NULL), 
          m_min_y(NULL), m_max_y(NULL),
          m_start_t(NULL), m_end_t(NULL),
          m_tid(NULL) {} 

    // Get a node id
    __host__ __device__ __forceinline__ 
    int get_node_id(int idx) const
    {
        return m_tid[idx];   
    }

    // Get a node bounding box
    __host__ __device__ __forceinline__
    float4 get_node_bbox(int idx) const
    {
        return make_float4(m_min_x[idx], m_min_y[idx], m_max_x[idx], m_max_y[idx]);
    }

    // Set a leaf node
    __host__ __device__ __forceinline__
    void set_node( int idx, const float4 &bbx, int id )
    {
        m_min_x[idx] = bbx.x;
        m_min_y[idx] = bbx.y;
        m_max_x[idx] = bbx.z;
        m_max_y[idx] = bbx.w;
        m_tid[idx]    =  id;
    }

    __host__ __device__ __forceinline__
    void set( float *min_x, float *min_y, 
              float *max_x, float *max_y, 
              long  *start, long  *end,
              int *id)
    {
        m_min_x     = min_x;
        m_min_y     = min_y;
        m_max_x     = max_x;
        m_max_y     = max_y;
        m_start_t   = start;
        m_end_t     = end;
        m_tid       = id;
    }   
};

////////////////////////////////////////////////////////////////////////////////
//// A 2D bounding box
//////////////////////////////////////////////////////////////////////////////////
//

class Bounding_box
{

    // Extreme points of the bounding box.
    float2 m_p_min;
    float2 m_p_max;
 
public:
    // Constructor. Create a unit box.
    __host__ __device__ Bounding_box()
    {
        m_p_min = make_float2( 0.0f, 0.0f );
        m_p_max = make_float2( 1.0f, 1.0f );
    }

    // Compute the center of the bounding-box.
    __host__ __device__ void compute_center( float2 &center ) const
    {
        center.x = 0.5f * ( m_p_min.x + m_p_max.x );
        center.y = 0.5f * ( m_p_min.y + m_p_max.y );
    }


  
    // The points of the box.
    __host__ __device__ __forceinline__ const float2& get_max() const
    {
        return m_p_max;
    }

    __host__ __device__ __forceinline__ const float2& get_min() const
    {
        return m_p_min;
    }

    // Does a tree box contain a leaf node.
    __host__ __device__ bool contains( const float4 &l ) const
    {
        // TODO bouning box interset
        return l.x >= m_p_min.x && l.z < m_p_max.x && l.y >= m_p_min.y && l.w < m_p_max.y;
    }

    // Does a tree box intersect a leaf node
    __host__ __device__ bool intersect( const float4 &l ) const
    {
        return m_p_min.x < l.z && m_p_max.x > l.x && m_p_max.y > l.y && m_p_min.y < l.w;
    }

    // Define the bounding box.
    __host__ __device__ void set( float min_x, float min_y, float max_x, float max_y )
    {
        m_p_min.x = min_x;
        m_p_min.y = min_y;
        m_p_max.x = max_x;
        m_p_max.y = max_y;
    }
};

////////////////////////////////////////////////////////////////////////////////
//// A node of a quadree.
//////////////////////////////////////////////////////////////////////////////////
class Quadtree_node
{
    // The identifier of the node
    int m_id;

    // The order in siblings
    int m_order;
    
    // The bounding box of the tree
    Bounding_box m_bbox;
    
    // The range of leaf nodes
    int m_begin, m_end; 

    // The child node
    Quadtree_node *m_children;

public:
    __host__ __device__ Quadtree_node() 
        : m_id(0), m_order(0), m_begin(0), m_end(0) {}

    // The ID of a node at its level.
    __host__ __device__ void set_id( int new_id )
    {
        m_id = new_id;
    }

    // The ID of a node at its level.
    
    __host__ __device__ int id() const
    {
        return m_id;
    }

    __host__ __device__ void set_order( int order )
    {
        m_order = order;
    }

    __host__ __device__ __forceinline__
    int order() const
    {
        return m_order;
    }

    // set current node's children
    __host__ __device__ __forceinline__
    void set_children( Quadtree_node *children )
    {
        m_children = children;
    }

    __host__ __device__ __forceinline__
    Quadtree_node * children()
    {
        return m_children;
    }
     
    // Set the bounding box.
    __host__ __device__ __forceinline__ 
    void set_bounding_box( float min_x, float min_y, float max_x, float max_y )
    { 
        m_bbox.set( min_x, min_y, max_x, max_y ); 
    }

    __host__ __device__ __forceinline__
    const Bounding_box& bounding_box() const
    {
        return m_bbox;
    }

    // The number of leaf nodes in the tree
    __host__ __device__ __forceinline__
    int num_points() const
    {
        return m_end - m_begin;
    }

    // the range of leaf nodes in the tree
    __host__ __device__ __forceinline__
    int leaf_begin() const
    {
        return m_begin;
    }

    __host__ __device__ __forceinline__
    int leaf_end() const
    {
        return m_end;
    }

    // Define the range for that node
    __host__ __device__ __forceinline__
    void set_range( int begin, int end )
    {
        m_begin = begin;
        m_end = end;
    }
};

////////////////////////////////////////////////////////////////////////////////
//// Algorithm parameters.
//////////////////////////////////////////////////////////////////////////////////
struct Parameters
{
    // Chose the right set of leaf nodes to use as in/out
    int leaf_selector;

    // The number of nodes at a given level (2^k for level k)
    int num_nodes_at_this_level;

    // The recursion depth
    int depth;

    // The max value for depth
    const int max_depth;

    // The minimum number of points in a node to stop recursion
    const int min_points_per_node;

    // Constructor set to default values.
    __host__ __device__ Parameters( int max_depth, int min_points_per_node ) : 
        leaf_selector(0), 
        num_nodes_at_this_level(1), 
        depth(0), 
        max_depth(max_depth), 
        min_points_per_node(min_points_per_node)
    {}
    
    // Copy constructor. Changes the values for next iteration.
    __host__ __device__ Parameters( const Parameters &params, bool ) :
        leaf_selector((params.leaf_selector+1) % 2), 
        num_nodes_at_this_level(4*params.num_nodes_at_this_level), 
        depth(params.depth+1), 
        max_depth(params.max_depth), 
        min_points_per_node(params.min_points_per_node)
    {}
};

class Query_result
{
    int * m_points_begin;   // store starting position of leaf_node vector
    int * m_points_end;     // store end position of leaf_node vector
    int * m_num;            // number of result

  public:
    
    // Add begin and end positions on query result
    __host__ __device__
    void set_values( int idx, int begin, int end )
    {
        m_points_begin[idx] = begin;
        m_points_end[idx] = end;   
    }

    // set values for member
    __host__ __device__
    void set( int * begin, int * end, int * num ) 
    {
        m_points_begin = begin;
        m_points_end = end;
        m_num = num;
    }

    // Get number of result
    __host__ __device__ int * num()
    {
        return m_num;
    }

    // Get begin and end values
    __host__ __device__ int * begin()
    {
        return m_points_begin;
    }

    __host__ __device__ int * end( )
    {
        return m_points_end;
    }
};

//Quadtree_node * d_nodes;

//Leaf_nodes * d_leaf;

//Query_result * d_query_result;

// traverse a path
void traverse( const char *, int );

// compute bounding box for each trajectory file
int compute_bbox( char *, float4 &, int2 & ) ;

// subdivide a trajectory's bounding box into segments w.r.t quadant rect
float4 * segment( float4 bbox, float4 rect);

// construct a quadtree from a trajectory data path
thrust::device_vector<int> construct_quadtree( const char * );

// given a rectangle region, search trajectory id on quadtree
void search( float * rect, long begin, long end, int * result, thrust::device_vector<int> );

#endif
