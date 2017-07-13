#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/count.h>
#include <thrust/set_operations.h>

#include <sys/stat.h>
#include <sys/types.h>
#include <dirent.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <math.h>

#include "parser.h"
#include "quadtree.h"
#include "helper_cuda.h"

char **         file_array;
int             num_file;
Quadtree_node * d_nodes;
Leaf_nodes *    d_leaf;
Query_result *  d_query_result;

// invoke before construct a quadtree
void init_file_array()
{
    num_file = 0;

    file_array = (char**) malloc(sizeof(char*) * 10000);
    for (int i = 0; i < 10000; ++i)
        file_array[i] = (char*) malloc( sizeof(char) * 100 );
}

template< int NUM_THREADS_PER_BLOCK >
__global__
void build_quadtree_kernel( Quadtree_node *nodes, Leaf_nodes *leaf,  Parameters params )
{
    // The number of warps in a block
    const int NUM_WARPS_PER_BLOCK = NUM_THREADS_PER_BLOCK / warpSize;

    // Shared memory to store the number of leaf nodes
    extern __shared__ int smem[];

    // Addresses of shared memory
    volatile int *s_num_lfs[4];
    for ( int i = 0; i < 4; ++i )
        s_num_lfs[i] = (volatile int *) &smem[i*NUM_WARPS_PER_BLOCK];

    // Compute the coordinates of the threads in the block
    const int warp_id = threadIdx.x / warpSize;
    const int lane_id = threadIdx.x % warpSize;

    // Mask for compaction
    int lane_mask_lt = (1 << lane_id) - 1;

    // The current node
    Quadtree_node &node = nodes[blockIdx.x];
    //node.set_id( node.id() + blockIdx.x );

    // The number of points in the node.
    int num_points = node.num_points();

    //   
    // 1- Check the number of points and its depth.
    //

    // Last step. Stop the recursion here. Make sure points[0] contains all the points.
    //printf("node points: %d\n", num_points);

    if( params.depth >= params.max_depth || num_points <= params.min_points_per_node )
    {
        if( params.leaf_selector == 1 )
        {
            int it = node.leaf_begin(), end = node.leaf_end();

            for( it += threadIdx.x ; it < end ; it += NUM_THREADS_PER_BLOCK )
                if( it < end )
                {
                    leaf[0].set_node( it, leaf[1].get_node_bbox( it ), leaf[1].get_node_id( it ) );
                    
                         //printf("---%d\n", leaf[1].get_node_id( it ) );
                }
        }
        return;
    }

    //printf("---------------%d\n", threadIdx.x);

    // Compute the center of the bounding box of the leaf nodes
    const Bounding_box &bbox = node.bounding_box();
    float2 center;
    bbox.compute_center( center );

    // Find how many leafs to give to each warp.
    int num_leaf_per_warp = max(warpSize, (num_points + NUM_WARPS_PER_BLOCK-1) / NUM_WARPS_PER_BLOCK);

    //Each warp of threads will compute the number of points to move to each quadrant
    int range_begin = node.leaf_begin() + warp_id * num_leaf_per_warp;
    int range_end   = min( range_begin + num_leaf_per_warp, node.leaf_end() );

    //
    // 2- Count the number of leafs in each child.
    //

    // Reset the counts of leafs per child
    if ( lane_id == 0 )
    {
        s_num_lfs[0][warp_id] = 0;
        s_num_lfs[1][warp_id] = 0;
        s_num_lfs[2][warp_id] = 0;
        s_num_lfs[3][warp_id] = 0;
    }

    // Input leafs
    const Leaf_nodes &in_leaf = leaf[params.leaf_selector];

    // Compute the number of leafs
    for (int range_it = range_begin + lane_id; __any(range_it < range_end); range_it += warpSize )
    {
        // Is it still an active thread?
        bool is_active = range_it < range_end;

        // Load the bounding box of the leaf node
        float4 bbx = is_active ? in_leaf.get_node_bbox( range_it ) : make_float4( 0.0f, 0.0f, 0.0f, 0.0f );

        // count top-left leafs
        int num_lfs = __popc( __ballot( is_active && bbx.x < center.x && bbx.w >= center.y ) );
        if ( num_lfs > 0 && lane_id == 0 )
            s_num_lfs[0][warp_id] += num_lfs;

        // count top-right leafs
        num_lfs = __popc( __ballot( is_active && bbx.z >= center.x && bbx.w >= center.y ) );
        if ( num_lfs > 0 && lane_id == 0 )
            s_num_lfs[1][warp_id] += num_lfs;

        // count bottom-left leafs
        num_lfs = __popc( __ballot( is_active && bbx.x < center.x && bbx.y < center.y ) );
        if ( num_lfs > 0 && lane_id == 0 )
            s_num_lfs[2][warp_id] += num_lfs;

        // count bottom-right leafs
        num_lfs = __popc( __ballot( is_active && bbx.z >= center.x && bbx.y < center.y ) );
        if ( num_lfs > 0 && lane_id == 0 )
            s_num_lfs[3][warp_id] += num_lfs;
    }

    // Make sure warps have finished counting
    __syncthreads();

    //
    // 3- Scan the warps' results to know the "global" numbers.
    //

    // First 4 warps scan the numbers of points per child (inclusive scan).
    if( warp_id < 4 )
    {
        int num_lfs = lane_id < NUM_WARPS_PER_BLOCK ? s_num_lfs[warp_id][lane_id] : 0;

       #pragma unroll
        for( int offset = 1 ; offset < NUM_WARPS_PER_BLOCK ; offset *= 2 )
        {
            int n = __shfl_up( num_lfs, offset, NUM_WARPS_PER_BLOCK );

            if( lane_id >= offset )
                num_lfs += n;
        }

        if( lane_id < NUM_WARPS_PER_BLOCK )
            s_num_lfs[warp_id][lane_id] = num_lfs;
    }

    __syncthreads();

    // Compute global offsets.
    if( warp_id == 0 )
    {
        int sum = s_num_lfs[0][NUM_WARPS_PER_BLOCK-1];
        for( int row = 1 ; row < 4 ; ++row )
        {
            int tmp = s_num_lfs[row][NUM_WARPS_PER_BLOCK-1];

            if( lane_id < NUM_WARPS_PER_BLOCK )
                s_num_lfs[row][lane_id] += sum;

            sum += tmp;
        }
    }

    __syncthreads();

    // Make the scan exclusive.
    if( threadIdx.x < 4*NUM_WARPS_PER_BLOCK )
    {
        int val = threadIdx.x == 0 ? 0 : smem[threadIdx.x-1];
        val += node.leaf_begin();
        smem[threadIdx.x] = val;
    }

    __syncthreads();

    //
    // 4- Move points.
    //

    // OUtput leaf nodes
    Leaf_nodes &out_leaf = leaf[(params.leaf_selector+1) % 2];

    // Reorder leaf nodes
    for ( int range_it = range_begin + lane_id; __any(range_it < range_end);  range_it += warpSize ) 
    {
        // Is it still an active thread?
        bool is_active = range_it < range_end;

        // Load the bounding box of leaf nodes
        float4 bbx = is_active ? in_leaf.get_node_bbox( range_it ) : make_float4( 0.0f, 0.0f, 0.0f, 0.0f );
        int tid = is_active ? in_leaf.get_node_id( range_it ) : -1;

        // Count top-left leaf nodes
        bool pred = is_active && bbx.x < center.x && bbx.w >= center.y;
        int vote = __ballot( pred );
        int dest = s_num_lfs[0][warp_id] + __popc( vote & lane_mask_lt );
        if ( pred ) out_leaf.set_node( dest, bbx, tid );
        if ( lane_id == 0 )
           s_num_lfs[0][warp_id] += __popc( vote );

        // count top-right leaf nodes
        pred = is_active && bbx.z >= center.x && bbx.w >= center.y;
        vote = __ballot( pred );
        dest = s_num_lfs[1][warp_id] + __popc( vote & lane_mask_lt );
        if ( pred ) out_leaf.set_node( dest, bbx, tid );
        if ( lane_id == 0 )
           s_num_lfs[1][warp_id] += __popc( vote );

        // count bottom-left leaf nodes
        pred = is_active && bbx.x < center.x && bbx.y < center.y;
        vote = __ballot( pred );
        dest = s_num_lfs[2][warp_id] + __popc( vote & lane_mask_lt );
        if ( pred ) out_leaf.set_node( dest, bbx, tid );
        if ( lane_id == 0 )
           s_num_lfs[2][warp_id] += __popc( vote );

        // Count bottom-right leaf nodes
        pred = is_active && bbx.z >= center.x && bbx.y < center.y;
        vote = __ballot( pred );
        dest = s_num_lfs[3][warp_id] + __popc( vote & lane_mask_lt );
        if ( pred ) out_leaf.set_node( dest, bbx, tid );
        if ( lane_id == 0 )
           s_num_lfs[3][warp_id] += __popc( vote );

    }

    __syncthreads();

    //
    // 5- Launch new blocks.
    //

    // The last thread launches new blocks.
    if( threadIdx.x == NUM_THREADS_PER_BLOCK-1 )
    {
        // The children
        Quadtree_node *children = &nodes[params.num_nodes_at_this_level];
     
        // The offsets of the children at their level
        int child_offset = 4*node.order();

            //printf("child offset: %d, blockIdx: %d\n", child_offset, blockIdx.x);
        int first_child_id = node.id() + params.num_nodes_at_this_level - node.order() + child_offset;

         // Set IDs
        children[child_offset+0].set_id( first_child_id + 0 );
        children[child_offset+1].set_id( first_child_id + 1 );
        children[child_offset+2].set_id( first_child_id + 2 );
        children[child_offset+3].set_id( first_child_id + 3 );

        // Set order in siblings
        children[child_offset+0].set_order( child_offset+ 0 );
        children[child_offset+1].set_order( child_offset+ 1 );
        children[child_offset+2].set_order( child_offset+ 2 );
        children[child_offset+3].set_order( child_offset+ 3 );

        // set parent's children
        node.set_children(&children[child_offset]);

        // bounding box
        const float2 &p_min = bbox.get_min();
        const float2 &p_max = bbox.get_max();


        // Set the bounding boxes of the children.
        children[child_offset+0].set_bounding_box( p_min.x , center.y, center.x, p_max.y  ); // Top-left.
        children[child_offset+1].set_bounding_box( center.x, center.y, p_max.x , p_max.y  ); // Top-right.
        children[child_offset+2].set_bounding_box( p_min.x , p_min.y , center.x, center.y ); // Bottom-left.
        children[child_offset+3].set_bounding_box( center.x, p_min.y , p_max.x , center.y ); // Bottom-right.

        // Set the ranges of the children.
        children[child_offset+0].set_range( node.leaf_begin(),     s_num_lfs[0][warp_id] );
        children[child_offset+1].set_range( s_num_lfs[0][warp_id], s_num_lfs[1][warp_id] ); 
        children[child_offset+2].set_range( s_num_lfs[1][warp_id], s_num_lfs[2][warp_id] ); 
        children[child_offset+3].set_range( s_num_lfs[2][warp_id], s_num_lfs[3][warp_id] ); 

        // Launch 4 children.
        build_quadtree_kernel<NUM_THREADS_PER_BLOCK>
        <<<
            4, 
            NUM_THREADS_PER_BLOCK, 
            4*NUM_WARPS_PER_BLOCK*sizeof(int)
        >>>
        ( &children[child_offset], leaf, Parameters( params, true ) );

    }

}


////////////////////////////////////////////////////////////////////////////////
// Make sure a Quadtree is properly defined.
////////////////////////////////////////////////////////////////////////////////
bool check_quadtree( const Quadtree_node *nodes, int idx, int num_pts, Leaf_nodes *pts, Parameters params )
{
    const Quadtree_node &node = nodes[idx];
    int num_points = node.num_points();

    if( params.depth == params.max_depth || num_points <= params.min_points_per_node )
    {
        int num_points_in_children = 0;

        num_points_in_children += nodes[params.num_nodes_at_this_level + 4*idx+0].num_points();
        num_points_in_children += nodes[params.num_nodes_at_this_level + 4*idx+1].num_points();
        num_points_in_children += nodes[params.num_nodes_at_this_level + 4*idx+2].num_points();
        num_points_in_children += nodes[params.num_nodes_at_this_level + 4*idx+3].num_points();

        if( num_points_in_children != node.num_points() )
            return false;

        return check_quadtree( &nodes[params.num_nodes_at_this_level], 
                    4*idx+0, num_pts, pts, Parameters( params, true ) ) 
            &&
            check_quadtree( &nodes[params.num_nodes_at_this_level], 
                    4*idx+1, num_pts, pts, Parameters( params, true ) ) 
            &&
            check_quadtree( &nodes[params.num_nodes_at_this_level], 
                    4*idx+2, num_pts, pts, Parameters( params, true ) ) 
            &&
            check_quadtree( &nodes[params.num_nodes_at_this_level], 
                    4*idx+3, num_pts, pts, Parameters( params, true ) );
    }
           

    const Bounding_box & bbox = node.bounding_box();

    for( int it = node.leaf_begin() ; it < node.leaf_end() ; ++it )
    {
        if( it >= num_pts )
            return false;

        float4 p = pts->get_node_bbox( it );

        if( !bbox.contains( p )  || !bbox.intersect( p ) )
            return false;
    }

    return true;
}

////////////////////////////////////////////////////////////////////////////////                                                        // Construct quadtree from a data folder
////////////////////////////////////////////////////////////////////////////////   

thrust::device_vector<int> construct_quadtree( const char *path ) 
{
    // Find/set the device
    int device_count = 0, device = -1, warp_size = 0;
    checkCudaErrors( cudaGetDeviceCount( &device_count ) );
    for ( int i = 0; i < device_count; ++i)
    {
        cudaDeviceProp properties;
        checkCudaErrors( cudaGetDeviceProperties( &properties, i ) );
        if ( properties.major > 3 || ( properties.major == 3 && properties.minor >= 5 ) )
        {
            device = i;
            warp_size = properties.warpSize;
            std::cout << "Running on GPU " << i << " (" << properties.name << ")" << std::endl;
            break;
        }
        std::cout << "GPU " << i << " (" << properties.name 
            << ") does not support CUDA Dynamic Parallelism" << std::endl;
    }

    if ( device == -1 )
    {
        std::cerr << "QuadTree requires SM 3.5 or higer to use CUDA Dynamic Parallelism. Exiting...\n" << std::endl;
        exit(EXIT_SUCCESS);
    }
    cudaSetDevice( device );

    // initlize file number 
    init_file_array();

    // read files in a path
    traverse( path, 0 );
    
    thrust::device_vector<float> min_x_vec(     num_file, -1 );
    thrust::device_vector<float> min_y_vec(     num_file, -1 ); 
    thrust::device_vector<float> max_x_vec(     num_file, -1 );
    thrust::device_vector<float> max_y_vec(     num_file, -1 );
    thrust::device_vector<long>  t_begin_vec(   num_file, -1 );
    thrust::device_vector<long>  t_end_vec(     num_file, -1 );
    thrust::device_vector<int>   tid_vec(       num_file, -1 );

    thrust::device_vector<float> min_x_vec_1(   num_file, -1 );
    thrust::device_vector<float> min_y_vec_1(   num_file, -1 ); 
    thrust::device_vector<float> max_x_vec_1(   num_file, -1 );
    thrust::device_vector<float> max_y_vec_1(   num_file, -1 );
    thrust::device_vector<long>  t_begin_vec_1( num_file, -1 );
    thrust::device_vector<long>  t_end_vec_1(   num_file, -1 );
    thrust::device_vector<int>   tid_vec_1(     num_file, -1 );


    // Host structrues 
    Leaf_nodes leaf_init[2];

    // compute bbox
    for ( int i = 0; i < num_file; ++i )
    {
        float4 bbx;
        int2 tid;

        compute_bbox( file_array[i], bbx, tid );
        
        min_x_vec[i] = bbx.x;
        min_y_vec[i] = bbx.y;
        max_x_vec[i] = bbx.z;
        max_y_vec[i] = bbx.w;
        tid_vec[i]   = tid.x;
    }

    //thrust::copy(max_x_vec.begin(), max_x_vec.end(), std::ostream_iterator<float>(std::cout, ","));

    leaf_init[0].set( thrust::raw_pointer_cast( &min_x_vec[0] ),
           thrust::raw_pointer_cast( &min_y_vec[0] ),
           thrust::raw_pointer_cast( &max_x_vec[0] ),
           thrust::raw_pointer_cast( &max_y_vec[0] ),
           thrust::raw_pointer_cast( &t_begin_vec[0] ),
           thrust::raw_pointer_cast( &t_end_vec[0] ),
           thrust::raw_pointer_cast( &tid_vec[0] ) );

    leaf_init[1].set( thrust::raw_pointer_cast( min_x_vec_1.data() ),
           thrust::raw_pointer_cast( min_y_vec_1.data() ),
           thrust::raw_pointer_cast( max_x_vec_1.data() ),
           thrust::raw_pointer_cast( max_y_vec_1.data() ),
           thrust::raw_pointer_cast( t_begin_vec.data() ),
           thrust::raw_pointer_cast( t_end_vec_1.data() ),
           thrust::raw_pointer_cast( tid_vec_1.data() ) );

    // Allocate memory  to stroe leaf nodes
    //Leaf_nodes *leaf;
    checkCudaErrors( cudaMalloc( (void**) &d_leaf, 2*sizeof(Leaf_nodes) ) );
    checkCudaErrors( cudaMemcpy( d_leaf, leaf_init, 2*sizeof(Leaf_nodes), cudaMemcpyHostToDevice ) );

    // Constants to control the algorithm
    const int num_leaf              = num_file;
    const int max_depth             = 3;
    const int min_points_per_node   = 1;

    // evaluate maximum number of nodes
    //int max_depth = log(num_leaf / min_points_per_node) / log(4);
    //int max_depth = 2;
    int max_nodes = pow(4, max_depth+1); 

    // Allocate memory to stroe the tree
    Quadtree_node root;
    root.set_range( 0, num_leaf );
    root.set_bounding_box( 120.459, 27.849, 120.947, 28.2);

    std::cout << "leaf number: " << num_leaf << " depth: " << max_depth << std::endl;
    //Quadtree_node * nodes;
    checkCudaErrors( cudaMalloc( (void**) &d_nodes, max_nodes*sizeof(Quadtree_node) ) );
    checkCudaErrors( cudaMemcpy( d_nodes, &root, sizeof(Quadtree_node), cudaMemcpyHostToDevice ) );

    // set the recursion limit for CDP to max_depth
    cudaDeviceSetLimit( cudaLimitDevRuntimeSyncDepth, max_depth );

    // Build the quadtree
    Parameters params( max_depth, min_points_per_node );
    std::cout << "Launching CDP kernel to build the quadtree" << std::endl;
    const int NUM_THREADS_PER_BLOCK = 128;  // Do not use less than 128 threads
    const int NUM_WARPS_PER_BLOCK = NUM_THREADS_PER_BLOCK / warp_size;
    const size_t smem_size = 4*NUM_WARPS_PER_BLOCK*sizeof(int);
    build_quadtree_kernel<NUM_THREADS_PER_BLOCK>
        <<<1, NUM_THREADS_PER_BLOCK, smem_size>>>
        ( d_nodes, d_leaf, params );
    checkCudaErrors( cudaGetLastError() );

    // Copy points to CPU
    thrust::host_vector<float> h_min_x_vec( min_x_vec );
    thrust::host_vector<float> h_min_y_vec( min_y_vec );
    thrust::host_vector<float> h_max_x_vec( max_x_vec );
    thrust::host_vector<float> h_max_y_vec( max_y_vec );
    thrust::host_vector<long>  h_begin_vec( 0 );
    thrust::host_vector<long>  h_end_vec  (0 );
    thrust::host_vector<int>   h_tid_vec(   tid_vec   );

    //thrust::copy( h_tid_vec.begin(), h_tid_vec.end(), std::ostream_iterator<int>(std::cout, ",") );

    Leaf_nodes host_leaf;
    host_leaf.set( thrust::raw_pointer_cast( h_min_x_vec.data() ),
           thrust::raw_pointer_cast( h_min_y_vec.data() ),
           thrust::raw_pointer_cast( h_max_x_vec.data() ),
           thrust::raw_pointer_cast( h_max_y_vec.data() ),
           thrust::raw_pointer_cast( h_begin_vec.data() ),
           thrust::raw_pointer_cast( h_end_vec.data() ),
           thrust::raw_pointer_cast( h_tid_vec.data() ));

    // Copy nodes to CPU
    Quadtree_node * host_nodes = new Quadtree_node[max_nodes];
    checkCudaErrors( cudaMemcpy( 
                host_nodes, 
                d_nodes, 
                max_nodes*sizeof(Quadtree_node), 
                cudaMemcpyDeviceToHost ) );

    for ( int i = 0; i < max_nodes; ++i )
    {
        std::cout << "node id1 " << host_nodes[i].id() << "\n";
        std::cout << "number points: " << host_nodes[i].num_points() << "\n";
    }

    // Validate the results
    //bool ok = check_quadtree( host_nodes, 0, num_leaf, &host_leaf, params );
    //std::cout << "Results: " << (ok ? "OK" : "FAILED") << std::endl;

    delete [] host_nodes;

    // Free memory
    //checkCudaErrors( cudaFree( nodes ) );
    
    return tid_vec;
    //exit( ok ? EXIT_SUCCESS : EXIT_FAILURE );

}

void traverse( const char * folder, int level )
{
    DIR * dir;
    struct dirent * ent;

    if ( ( dir = opendir(folder) ) != NULL )
    {
        while ( ( ent = readdir(dir) ) != NULL )
        {
            if ( ent->d_type == DT_DIR )
            {
                char path[100];
                int len = snprintf( path, sizeof(path)-1, "%s/%s", folder, ent->d_name );
                path[len] = 0;

                // skip self and parent
                if ( strcmp(ent->d_name, ".") == 0 || strcmp(ent->d_name, "..") == 0 )
                    continue;

                if ( strlen(folder)+strlen(ent->d_name)+2 > sizeof(path) )
                    fprintf( stderr, "name %s %s too long\n", folder, ent->d_name );

                traverse( path, level+1 );
            }
            else
            {
                char file_name[200];
                sprintf( file_name, "%s/%s", folder, ent->d_name );
                strcpy( file_array[num_file++], file_name );
            }
        }

        closedir( dir );
    }    
    else
    {
        perror( "read file error!" );
        return;
    }
} 

int compute_bbox( char * filename, float4 & bbox, int2 & tid )
{
    // get file size and prepare to map file
    FILE * f = fopen( filename, "r" );
    if ( f == NULL )
    {
        perror( "File name not exists!" );
        return EXIT_FAILURE;
    }

    fseek( f, 0, SEEK_END );
    unsigned long file_size = ftell( f );
    thrust::device_vector<char> dev( file_size );
    fclose( f );

    // map file from disk to memory
    char * p_mmap;
    int fd = open( filename, O_RDONLY | O_NONBLOCK );
    if ( fd == -1 )
    {
        perror ("open file error!" );
        return EXIT_FAILURE;
    }

    if ( file_size > 0 )
        p_mmap = (char *) mmap( 0, file_size, PROT_READ, MAP_SHARED, fd, 0 );

    if ( p_mmap == MAP_FAILED )
    {
        perror( "mmap error!" );
        return EXIT_FAILURE;
    }

    if ( close(fd) == -1 )
    {
        perror( "close" );
        return EXIT_FAILURE;
    }

    // copy data from memory to GPU
    thrust::copy( p_mmap, p_mmap+file_size, dev.begin() );
            
    // count the lines of the file
    int cnt = thrust::count(dev.begin(), dev.end(), '\n');
    std::cout << "There are " << cnt << " total lines in the file" << std::endl;

    // find the position of "\n" to locate each line
    thrust::device_vector<int> line_index( cnt + 1 );
    line_index[0] = -1;

    // find the starting position from data 
    thrust::copy_if(
            thrust::make_counting_iterator( (unsigned int) 0 ),
            thrust::make_counting_iterator( (unsigned int) file_size ),
            dev.begin(),
            line_index.begin()+1,
            is_break() );

    // initialize column vectors
    thrust::device_vector<char> id_vec(         cnt*4  );
    thrust::device_vector<char> longitude_vec(  cnt*9  );
    thrust::device_vector<char> latitude_vec(   cnt*8  );
    thrust::device_vector<char> time_vec(       cnt*23 );
    thrust::device_vector<char> state_vec(      cnt*1  );
    thrust::device_vector<char> speed_vec(      cnt*2  );
    thrust::device_vector<char> direction_vec(  cnt*1  );

    thrust::device_vector<char*> dest( 7 );
    dest[0] = thrust::raw_pointer_cast( id_vec.data() );
    dest[1] = thrust::raw_pointer_cast( longitude_vec.data() );
    dest[2] = thrust::raw_pointer_cast( latitude_vec.data() );
    dest[3] = thrust::raw_pointer_cast( time_vec.data() );
    dest[4] = thrust::raw_pointer_cast( state_vec.data() );
    dest[5] = thrust::raw_pointer_cast( speed_vec.data() );
    dest[6]= thrust::raw_pointer_cast( direction_vec.data() );

    // set field max length
    thrust::device_vector<unsigned int> dest_len( 7 );
    dest_len[0] = 4;
    dest_len[1] = 9;
    dest_len[2] = 8;
    dest_len[3] = 23;
    dest_len[4] = 1;
    dest_len[5] = 2;
    dest_len[6] = 1;

    // set index for each column, default 7 columns
    thrust::device_vector<unsigned int> index( 7 );
    thrust::sequence( index.begin(), index.end() );

    thrust::device_vector<unsigned int> index_cnt(1);  // fields count
    index_cnt[0] = 7;

    thrust::device_vector<char> sep(1);
    sep[0] = ',';

                 
    thrust::counting_iterator<unsigned int> begin(0);   
    parser parse((const char*)thrust::raw_pointer_cast(dev.data()), 
            (char**)thrust::raw_pointer_cast(dest.data()),
            thrust::raw_pointer_cast(index.data()), 
            thrust::raw_pointer_cast(index_cnt.data()),
            thrust::raw_pointer_cast(sep.data()),
            thrust::raw_pointer_cast(line_index.data()),
            thrust::raw_pointer_cast(dest_len.data()));
    thrust::for_each(begin, begin + cnt, parse);

    // initialize database column container
    thrust::device_vector<int> _id_vec( cnt );
    thrust::device_vector<double> _lon_vec( cnt );
    thrust::device_vector<double> _lat_vec( cnt );

    // parse id column
    index_cnt[0] = 4;
    gpu_atoi atoi_id((const char*)thrust::raw_pointer_cast( id_vec.data() ),
            (int *)thrust::raw_pointer_cast( _id_vec.data() ),
            thrust::raw_pointer_cast( index_cnt.data() ));
    thrust::for_each( begin, begin + cnt, atoi_id );

    // parse longitude column
    index_cnt[0] = 9;
    gpu_atof atof_ff_lon((const char *)thrust::raw_pointer_cast( longitude_vec.data() ),
            (double *)thrust::raw_pointer_cast( _lon_vec.data() ),
            thrust::raw_pointer_cast( index_cnt.data() ));
    thrust::for_each( begin, begin + cnt, atof_ff_lon);

    // parse longitude column
    index_cnt[0] = 8;
    gpu_atof atof_ff_lat((const char *)thrust::raw_pointer_cast( latitude_vec.data() ),
            (double *)thrust::raw_pointer_cast( _lat_vec.data() ),
            thrust::raw_pointer_cast( index_cnt.data() ));
    thrust::for_each( begin, begin + cnt, atof_ff_lat );

    // sort coordinates to get bounding box
    thrust::sort_by_key( _lon_vec.begin(), _lon_vec.end(), _lat_vec.begin() );
    bbox = make_float4( _lon_vec[0], _lat_vec[0], _lon_vec[cnt-1], _lat_vec[cnt-1]);
    tid = make_int2( _id_vec[0], 0 );
  
    //std::cout << tid.x << std::endl;

    free( filename );

    return 1;
} 

template < int NUM_THREADS_PER_BLOCK >
__global__
float4 * segment( float *x, float *y, float4 rect )
{
    // if x not outside of rect, find the intersection
    if ( ! (x[threadIdx.x] < rect.x && x[threadIdx.x+1] < rect.x) 
            && ! (x[threadIdx.x] > rect.x && x[threadIdx.x+1] > rect.x) )     
    {
        
    }
}

// input a rectangle area and output a series of trajectory id
template < int NUM_THREADS_PER_BLOCK >
__global__
void search_kernel( 
        float4 rect, 
        long begin_t,
        long end_t, 
        Query_result * result,
        Quadtree_node * nodes, 
        Parameters params )
{
    Quadtree_node &node = nodes[blockIdx.x];

    // traverse to the max depth
    if ( params.depth >= params.max_depth && node.id() <= pow(4, params.max_depth) && threadIdx.x == 0 ) 
    {
        int *num = result->num();
        register int idx = atomicAdd( num, 1 );

        printf("---idx:--%d, leaf begin: %d, leaf end: %d \n", node.id(), node.leaf_begin(), node.leaf_end() );
        result->set_values( idx, node.leaf_begin(), node.leaf_end() );
        return;    
    }

    // The last thread launches new blocks
    if ( params.depth < params.max_depth && threadIdx.x == NUM_THREADS_PER_BLOCK-1 )
    {
        // if node intersect rect
        Bounding_box bbox = node.bounding_box();
        if ( !bbox.intersect( rect ) ) return;

        //Quadtree_node *children = node.children();
        Quadtree_node *children = &nodes[params.num_nodes_at_this_level];

        // The offsets of the children at their level
        int child_offset = 4*node.order();

        search_kernel<NUM_THREADS_PER_BLOCK>
            <<< 4, NUM_THREADS_PER_BLOCK>>>
            ( rect, begin_t, end_t, result, &children[child_offset], Parameters( params, true) );
    }

}

void unfold( thrust::device_vector<int> d_begin, thrust::device_vector<int> d_end, thrust::device_vector<int> d_tid )
{
    for (int i = 0; i < d_begin.size(); i++ )
    {
        // get trajectory id from leaf nodes
       // if ( h_begin[i] > 100 ) continue;

       for ( int j = d_begin[i]; j < d_end[i]; j++ )
        {
            //int tid = d_tid[j-1];
            printf("id: %d\n", j);
        }

        //thrust::copy( d_leaf+d_begin[i], d_leaf+d_end[i], d_temp_1.begin() );
        //auto p_end = thrust::set_intersection(d_temp_1.begin(), d_temp_1.begin()+d_end[i]-d_begin[i], 
        //                d_result_id.begin(), d_result_id.end(), d_temp_2.begin() );
        //thrust::copy( d_temp_2.begin(), p_end, d_result_id.begin() );
    }
    
}

void search( float *rect, long begin, long end, int * result, thrust::device_vector<int> tid_vec )
{
    int max_depth = 3;

    Parameters params( max_depth, 1 );

    // initialize query result
    thrust::device_vector<int> d_begin( num_file );
    thrust::device_vector<int> d_end(   num_file );
    thrust::device_vector<int> d_num(   1        );
    thrust::device_vector<int> d_result_id( num_file );     // store all id of result

    Query_result h_query_result;
    h_query_result.set( 
            thrust::raw_pointer_cast( &d_begin[0] ),
            thrust::raw_pointer_cast( &d_end[0] ), 
            thrust::raw_pointer_cast( &d_num[0] ) );
    
    //Query_result *d_query_result;
    checkCudaErrors( cudaMalloc( (void**) &d_query_result, sizeof(Query_result) ) );
    checkCudaErrors( cudaMemcpy( 
                        d_query_result, 
                        &h_query_result, 
                        sizeof(Query_result), 
                        cudaMemcpyHostToDevice ) );

    std::cout << "Launching search kernel to query trajectory " << std::endl;
    const int NUM_THREADS_PER_BLOCK = 128;
    
    float4 d_rect = make_float4( rect[0], rect[1], rect[2], rect[3] );
    search_kernel<NUM_THREADS_PER_BLOCK>
        <<<1, NUM_THREADS_PER_BLOCK>>>
        ( d_rect, begin, end, d_query_result, d_nodes, params);
    checkCudaErrors( cudaGetLastError() );
    
    // unfold query result to pave trajectory id
    unfold(d_begin, d_end, tid_vec );
        
    // check whether trajectories in the time span
    //d_leaf      
    /*int idx = 0;
    for ( int i = 0; i < *d_query_result->num(); ++i )
    {
        int it = d_query_result->begin(i), end = d_query_result->end(i);
        while ( it++ < end )
        {
            d_result[idx++] = d_leaf->get_node_id( it );
        }  
    }*/

    checkCudaErrors( cudaFree( d_nodes ) );
    checkCudaErrors( cudaFree( d_query_result ) );

    //checkCudaErrors( cudaMemcpy( result, d_result, (idx-1)*sizeof(int), cudaMemcpyDeviceToHost ) );
    //checkCudaErrors( cudaGetLastError() );
}
