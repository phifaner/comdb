#ifndef R_TREE_H
#define R_TREE_H

#include <stdlib.h>
#include <vector>

#include "s2/base/integral_types.h"

#define DIV_CEIL(x, y) (((x) + (y) - 1) / (y))

struct RTree_Point 
{
    uint64          x;
    unsigned long   y;
    int             id;

    bool operator==(RTree_Point p)
    {
        if (p.id == id) return true; 
        return false;  
    }
};

struct RTree_Points
{
    uint64          *X;
    unsigned long   *Y;
    int             *ID;
    size_t          length;
};

#define RTREE_NODE_SIZE         4
//#define MAX_THREADS_PER_BLOCK   100

struct RTree_Node;
struct RTree
{
    size_t          depth;   // know where the leaves begin
    RTree_Node   *  root;
    size_t          num;    // node number
};

struct RTree_Rect
{
    uint64 left, right;
    unsigned long top, bottom;
};

struct RTree_Leaf
{
    RTree_Rect      bbox;
    size_t          num;        // number of children
    size_t          depth;      // level
    RTree_Point *   points;
        
};

struct RTree_Node
{
    RTree_Rect      bbox;
    size_t          num;
    size_t          depth;
    RTree_Node *    children;
};

struct CUDA_RTree_Node
{
    int starting;
    int num_edges;
};

//struct CUDA_RTree_Node *d_nodes;
//int *d_edeges;
//struct RTree_Rect *d_rects;



void cuda_sort(RTree_Points *points);

RTree_Leaf* cuda_create_leaves(RTree_Points *sorted);

RTree_Node* cuda_create_level(RTree_Node *nodes, const size_t len, size_t depth);

RTree cuda_create_rtree(RTree_Points points);

int cpu_search(RTree_Node *N, RTree_Rect *rect, std::vector<int> &points);

RTree_Points cuda_search(RTree *tree, std::vector<RTree_Rect> rect_vec);

#endif
