#include "rtree.h"

#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>

void test_cuda_sort()
{
    uint64 h_x[10] = {2, 4, 5, 9, 8, 1, 7, 6, 3, 11};
    unsigned long h_y[10] = {2, 4, 5, 9, 8, 1, 7, 6, 3, 11};
    int h_id[10] = {2, 4, 5, 9, 8, 1, 7, 6, 3, 11};
    thrust::device_vector<uint64> d_x(h_x, h_x+10);
    thrust::device_vector<unsigned long> d_y(h_y, h_y+10);
    thrust::device_vector<int> d_id(h_id, h_id+10);
    
    thrust::copy(d_id.begin(), d_id.end(), std::ostream_iterator<int>(std::cout, ","));

    printf("=================\n");

    RTree_Points points;
    points.X = thrust::raw_pointer_cast(d_x.data());
    points.Y = thrust::raw_pointer_cast(d_y.data());
    points.ID = thrust::raw_pointer_cast(d_id.data());
    points.length = 10;

    //uint64 *X = thrust::raw_pointer_cast(d_x.data());
    
    //thrust::copy(d_id.begin(), d_id.end(), std::ostream_iterator<int>(std::cout, ","));

    //unsigned long *Y = thrust::raw_pointer_cast(d_y.data());
    //int *ID =  thrust::raw_pointer_cast(d_id.data());
    auto tbegin = thrust::make_zip_iterator(thrust::make_tuple(points.Y, points.ID));
    auto tend = thrust::make_zip_iterator(thrust::make_tuple(points.Y+10, points.ID+10));
    
    //thrust::copy(d_id.begin(), d_id.end(), std::ostream_iterator<int>(std::cout, ","));

    //thrust::sort_by_key(thrust::device, X, X+10, tbegin);


    cuda_sort(&points);

    thrust::copy(d_x.begin(), d_x.end(), std::ostream_iterator<uint64>(std::cout, ","));
}

void test_cuda_create_leaves()
{
    uint64 h_x[10] = {2, 4, 5, 9, 8, 1, 7, 6, 3, 11};
    unsigned long h_y[10] = {2, 4, 5, 9, 8, 1, 7, 6, 3, 11};
    int h_id[10] = {2, 4, 5, 9, 8, 1, 7, 6, 3, 11};
    thrust::device_vector<uint64> d_x(h_x, h_x+10);
    thrust::device_vector<unsigned long> d_y(h_y, h_y+10);
    thrust::device_vector<int> d_id(h_id, h_id+10);
   
    RTree_Points points;
    points.X = thrust::raw_pointer_cast(d_x.data());
    points.Y = thrust::raw_pointer_cast(d_y.data());
    points.ID = thrust::raw_pointer_cast(d_id.data());
    points.length = 10;

    cuda_sort(&points);
    
    RTree_Leaf *leaves = cuda_create_leaves(&points);
    
    // print leaves
    const size_t len_leaf = DIV_CEIL(points.length, RTREE_NODE_SIZE);
    for (size_t i = 0, end = len_leaf; i != end; ++i)
        //printf("depth: %lu, MBR: %llu, %llu, %lu, %lu\n", leaves[i].depth, leaves[i].bbox.left, leaves[i].bbox.right, leaves[i].bbox.bottom, leaves[i].bbox.top);
        for (size_t j = 0; j < leaves[i].num; j++)
            printf("id %lu: %llu\n", i, leaves[i].points[j].x);
}

void test_cuda_create_level()
{
    uint64 h_x[10] = {2, 4, 5, 9, 8, 1, 7, 6, 3, 11};
    unsigned long h_y[10] = {2, 4, 5, 9, 8, 1, 7, 6, 3, 11};
    int h_id[10] = {2, 4, 5, 9, 8, 1, 7, 6, 3, 11};
    thrust::device_vector<uint64> d_x(h_x, h_x+10);
    thrust::device_vector<unsigned long> d_y(h_y, h_y+10);
    thrust::device_vector<int> d_id(h_id, h_id+10);
   
    RTree_Points points;
    points.X = thrust::raw_pointer_cast(d_x.data());
    points.Y = thrust::raw_pointer_cast(d_y.data());
    points.ID = thrust::raw_pointer_cast(d_id.data());
    points.length = 10;

    cuda_sort(&points);
    
    RTree_Leaf *leaves = cuda_create_leaves(&points);
    const size_t len_leaf = DIV_CEIL(points.length, RTREE_NODE_SIZE);

    RTree_Node *previous_level = (RTree_Node*) leaves;
    size_t previous_len = len_leaf;

    RTree_Node *nodes = cuda_create_level(previous_level, previous_len, 1);

    const size_t len_level = DIV_CEIL(previous_len, RTREE_NODE_SIZE);
    for (size_t i = 0, end = len_level; i != end; ++i)
        printf("MBR: %llu, %llu, %lu, %lu\n", nodes[i].bbox.left, nodes[i].bbox.right, nodes[i].bbox.bottom, nodes[i].bbox.top);
}

void test_cuda_create_rtree()
{
    uint64 h_x[10] = {2, 4, 5, 9, 8, 1, 7, 6, 3, 11};
    unsigned long h_y[10] = {2, 4, 5, 9, 8, 1, 7, 6, 3, 11};
    int h_id[10] = {2, 4, 5, 9, 8, 1, 7, 6, 3, 11};
    thrust::device_vector<uint64> d_x(h_x, h_x+10);
    thrust::device_vector<unsigned long> d_y(h_y, h_y+10);
    thrust::device_vector<int> d_id(h_id, h_id+10);
   
    RTree_Points points;
    points.X = thrust::raw_pointer_cast(d_x.data());
    points.Y = thrust::raw_pointer_cast(d_y.data());
    points.ID = thrust::raw_pointer_cast(d_id.data());
    points.length = 10;

    RTree tree = cuda_create_rtree(points);
    
    printf("depth: %lu, bbox: %llu, %llu, %lu, %lu\n", tree.depth, tree.root->bbox.left, tree.root->bbox.right,
         tree.root->bbox.top, tree.root->bbox.bottom );

}

void test_cpu_search()
{
    uint64 h_x[10] = {2, 4, 5, 9, 8, 1, 7, 6, 3, 11};
    unsigned long h_y[10] = {2, 4, 5, 9, 8, 1, 7, 6, 3, 11};
    int h_id[10] = {2, 4, 5, 9, 8, 1, 7, 6, 3, 11};
    thrust::device_vector<uint64> d_x(h_x, h_x+10);
    thrust::device_vector<unsigned long> d_y(h_y, h_y+10);
    thrust::device_vector<int> d_id(h_id, h_id+10);
   
    RTree_Points points;
    points.X = thrust::raw_pointer_cast(d_x.data());
    points.Y = thrust::raw_pointer_cast(d_y.data());
    points.ID = thrust::raw_pointer_cast(d_id.data());
    points.length = 10;

    RTree tree = cuda_create_rtree(points);

    RTree_Rect R = {5, 9, 3, 5};
    //cpu_search(tree.root, &R);   
}

void test_cuda_search()
{
     uint64 h_x[10] = {2, 4, 5, 9, 8, 1, 7, 6, 3, 11};
    unsigned long h_y[10] = {2, 4, 5, 9, 8, 1, 7, 6, 3, 11};
    int h_id[10] = {2, 4, 5, 9, 8, 1, 7, 6, 3, 11};
    thrust::device_vector<uint64> d_x(h_x, h_x+10);
    thrust::device_vector<unsigned long> d_y(h_y, h_y+10);
    thrust::device_vector<int> d_id(h_id, h_id+10);
   
    RTree_Points points;
    points.X = thrust::raw_pointer_cast(d_x.data());
    points.Y = thrust::raw_pointer_cast(d_y.data());
    points.ID = thrust::raw_pointer_cast(d_id.data());
    points.length = 10;

    RTree tree = cuda_create_rtree(points);

    RTree_Rect R = {5, 9, 3, 5};

    std::vector<RTree_Rect> rect_vec;
    rect_vec.push_back(R);

    cuda_search(&tree, rect_vec);

}


int main()
{
    //test_cuda_sort();
    //test_cuda_create_leaves();
    //test_cuda_create_level();
    //test_cuda_create_rtree();
    //test_cpu_search();
    test_cuda_search();

}
