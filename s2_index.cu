#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <fcntl.h>

#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

#include <ctime>

#include "s2_index.h"
#include "parser.h"
#include "rtree.h"
#include "construct.h"
#include "Trie.h"

RTree Movix::build_index_rtree(const char *path)
{
    //char *filename = "/home/wangfei/201401TRK/TRK20140101/C02/C02668_n.txt";
    Construct c;
    int level = c.find_files(path);
    char ** f_array = c.get_file_array();
    int f_num = c.get_file_num();

    for (int i = 0; i < f_num; i++)
    {   
        printf("read index from : %s\n", f_array[i]); 
        // read cell points from file
        read_cell_points(f_array[i]);

    }

    // construct RTree from cell points
    /*thrust::sort_by_key(cell_vec.begin(), cell_vec.end(), 
               thrust::make_zip_iterator(thrust::make_tuple(ts_vec.begin(), tid_vec.begin())) );
 
    thrust::copy(cell_vec.begin(), cell_vec.end(), std::ostream_iterator<unsigned long long>(std::cout, ","));
    */
 
    RTree_Points points;
    points.X    = thrust::raw_pointer_cast(cell_vec.data());
    points.Y    = thrust::raw_pointer_cast(ts_vec.data());
    points.ID   = thrust::raw_pointer_cast(tid_vec.data());
    points.length = length;

    return cuda_create_rtree(points);

}

HMix Movix::build_bitmap_index(const char *path)
{

    // read a trajectory path
    Construct c;
    int level = c.find_files(path);
    char ** f_array = c.get_file_array();
    int f_num = c.get_file_num();

    // read each trajectory and then create index
    for (int i = 0; i < f_num; ++i)
    {
        read_cell_points(f_array[i]);
    }

    // copy all trajectory points to Host
    H_Points points;
    points.X        = (uint64 *) malloc(length * sizeof(uint64));
    points.Y        = (unsigned long *) malloc(length * sizeof(unsigned long));
    points.ID       = (int *) malloc(length * sizeof(int));
    points.length   = length;



    // order points on GPU by tid
    // thrust::sort_by_key( tid_vec.begin(), tid_vec.end(), 
    //     thrust::make_zip_iterator(thrust::make_tuple(ts_vec.begin(), cell_vec.begin())) );
 
    // thrust::copy(tid_vec.begin(), tid_vec.end(), std::ostream_iterator<int>(std::cout, ","));

    cudaMemcpy(points.X, 
            thrust::raw_pointer_cast(cell_vec.data()), 
            length*sizeof(uint64), cudaMemcpyDeviceToHost);

    cudaMemcpy(points.Y, 
            thrust::raw_pointer_cast(ts_vec.data()), 
            length*sizeof(unsigned long), cudaMemcpyDeviceToHost);

    cudaMemcpy(points.ID,
            thrust::raw_pointer_cast(tid_vec.data()),
            length*sizeof(int), cudaMemcpyDeviceToHost);

    // create indexes
    HMix H;
    H.hindex_create(points, level);

    std::cout << "create indexes finished" << std::endl;
    return H;
}

void Movix::cuda_build_bitmap_index(const char *path)
{
    // read a trajectory path
    Construct c;
    int level = c.find_files(path);
    char ** f_array = c.get_file_array();
    int f_num = c.get_file_num();

    // read each trajectory and then create index
    for (int i = 0; i < f_num; ++i)
    {
        read_cell_points(f_array[i]);
    }

    // order points on GPU by tid
    thrust::sort_by_key( tid_vec.begin(), tid_vec.end(), 
        thrust::make_zip_iterator(thrust::make_tuple(ts_vec.begin(), cell_vec.begin())) );

    // copy all trajectory points to Host
    // H_Points points;
    // points.X        = thrust::raw_pointer_cast(cell_vec.data());
    // points.Y        = thrust::raw_pointer_cast(ts_vec.data());
    // points.ID       = thrust::raw_pointer_cast(tid_vec.data());
    // points.length   = length;

// thrust::copy(tid_vec.begin(), tid_vec.end(), std::ostream_iterator<int>(std::cout, ","));

    // return points;
}

int Movix::read_cell_points(char * filename)
{
    // set device no
    cudaSetDevice(2);

    // get file size and prepare to map file
    FILE * f = fopen(filename, "r");
    if (f == NULL) 
    {
        perror("File name not exists!");
        return EXIT_FAILURE;
    }

    fseek(f, 0, SEEK_END);
    unsigned long file_size = ftell(f);
    thrust::device_vector<char> dev(file_size);
    fclose(f);

    // map file from disk to memory
    char * p_mmap;
    int fd = open(filename, O_RDONLY);
    if (fd == -1)
    {
        perror("Open file error");
        return EXIT_FAILURE;
    }

    if (file_size > 0)
        p_mmap = (char *) mmap(0, file_size, PROT_READ, MAP_SHARED, fd, 0);

    if (p_mmap == MAP_FAILED)
    {
        perror("mmap error!");
        return EXIT_FAILURE;
    }

    if(close(fd) == -1)
    {
        perror("close");
        return EXIT_FAILURE;
    }

    // copy data from memory to GPU
    thrust::copy(p_mmap, p_mmap+file_size, dev.begin());
    
    // count the lines of the data
    int cnt = thrust::count(dev.begin(), dev.end(), '\n');
    
    // find the position of '\n' to locate each line
    thrust::device_vector<int> line_index(cnt+1);
    line_index[0] = -1;
   
    // the following line is very important
    // to get the line starting position in the data
    thrust::copy_if(thrust::make_counting_iterator((unsigned int)0),
            thrust::make_counting_iterator((unsigned int) file_size),
            dev.begin(), line_index.begin()+1, is_break());

    // initialize column vectors
    thrust::device_vector<char> c_cid_vec(cnt*19, 0);
    thrust::device_vector<char> c_tid_vec(cnt*5);
    thrust::device_vector<char> c_time_vec(cnt*10);

    thrust::device_vector<char*> dest(3);
    dest[0] = thrust::raw_pointer_cast(c_cid_vec.data());
    dest[1] = thrust::raw_pointer_cast(c_tid_vec.data());
    dest[2] = thrust::raw_pointer_cast(c_time_vec.data());

    // set field max length
    thrust::device_vector<unsigned int> dest_len(3);
    dest_len[0] = 19;
    dest_len[1] = 5;
    dest_len[2] = 10;

    // set index for each column, default 3 columns
    thrust::device_vector<unsigned int> index(3);
    thrust::sequence(index.begin(), index.end());

    // set fields count
    thrust::device_vector<unsigned int> index_cnt(1);
    index_cnt[0] = 3;
    
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
    thrust::for_each(begin, begin+cnt, parse);

    // initialize containers for the current trajectory
    thrust::device_vector<int> _tid_vec(cnt);
    thrust::device_vector<uint64> _cell_vec(cnt);
    thrust::device_vector<unsigned long> _ts_vec(cnt);
    
    // resize container for all trajectories
    length = tid_vec.size() + cnt;
    try
    {
        tid_vec.resize(length);
        cell_vec.resize(length);
        ts_vec.resize(length);
    }
    catch (std::length_error)
    {
        printf("------------resize exceed Max Length! ---------\n");
    }

    // parse cell id column
    index_cnt[0] = 19;
    gpu_atoull atoll_cid((const char*)thrust::raw_pointer_cast(c_cid_vec.data()),
            (unsigned long long*)thrust::raw_pointer_cast(_cell_vec.data()),
            thrust::raw_pointer_cast(index_cnt.data()));
    thrust::for_each(begin, begin+cnt, atoll_cid);

    // parse trajectory id column
    index_cnt[0] = 5;
    gpu_atoi atoi_tid((const char*)thrust::raw_pointer_cast(c_tid_vec.data()),
            (int*)thrust::raw_pointer_cast(_tid_vec.data()),
            thrust::raw_pointer_cast(index_cnt.data()));
    thrust::for_each(begin, begin+cnt, atoi_tid);

    // parse timestamp column
    index_cnt[0] = 10;
    gpu_atoul atoul_ts((const char*)thrust::raw_pointer_cast(c_time_vec.data()),
            (unsigned long*)thrust::raw_pointer_cast(_ts_vec.data()),
            thrust::raw_pointer_cast(index_cnt.data()));
    thrust::for_each(begin, begin+cnt, atoul_ts);
    
    // copy new items into containers
    thrust::copy(_cell_vec.begin(),  _cell_vec.end(),   cell_vec.end() - cnt);
    thrust::copy(_ts_vec.begin(),   _ts_vec.end(),      ts_vec.end() - cnt);
    thrust::copy(_tid_vec.begin(),  _tid_vec.end(),     tid_vec.end() - cnt);

    // thrust::copy(tid_vec.begin(), tid_vec.end(), std::ostream_iterator<unsigned long long>(std::cout, ","));

    return level;
}

unsigned long get_time_interval(uint64_t * cells, size_t len, std::map<uint64, unsigned long> map)
{
    unsigned long ts_min = 9999999999999999l, ts_max = 0;

    for (size_t i = 0; i < len; ++i)
    {
        std::map<uint64, unsigned long>::iterator it = map.find(cells[i]);

        // found
        if (it != map.end())
        {
            if (ts_max < it->second) ts_max = it->second;
            if (ts_min > it->second) ts_min = it->second; 
        }
    }

    assert(ts_max != 0);
    assert(ts_min != 9999999999999999l);

    return ts_max - ts_min;
}

std::map<int, H_Index>
Movix::cost_by_time(HMix *H, std::vector<std::vector<uint64> > V, unsigned long *T, double v)
{
    // different types of POI
    assert (!V.empty());
    assert (T[1] >= T[0]);

    // from small to big 
    std::sort(V.begin(), V.end(), [](const std::vector<uint64> & a, const std::vector<uint64> & b) -> bool {  return a.size() > b.size(); });

    // counting intersecting trajectories with POI in [s, e]
    std::map<int, std::vector<uint64> > t_map;
    //size_t dist_count = 0;       
    Trie *p_t = new Trie(v, (T[1]-T[0]));
    std::vector<Edge> edges;

    // get the first meaningful type vector
    std::vector<uint64> type_1_cells = V.back();
    V.pop_back();
    while ( type_1_cells.size() == 0)
    {
        type_1_cells = V.back();
        V.pop_back();
    }

    // transform POI into Edge
    std::vector<uint64>::iterator tit, tend;
    for (tit = type_1_cells.begin(), tend = type_1_cells.end(); tit != tend; ++tit)
    {
        Edge eg;
        eg.cells.push_back(*tit);
        eg.distance = 0;
        edges.push_back(eg);
    }

    printf("edge size: %lu\n", V[1].size());

    std::map<int, H_Index> hmap = H->get_hmap();

    printf("-----------bitmap index size:%lu\n", hmap.size());

    std::map<int, H_Index> v_map = hmap;

    // if the distance between two cells is larger than t*voc
    while ( !V.empty() )
    {
        std::map<int, H_Index> mid_map;

        std::vector<uint64> type_cells = V.back();
        V.pop_back();

        edges = p_t->multix( edges, type_cells );

        printf("------after edge size: %lu\n", edges.size());

        // construct bitmaps from edges
        std::vector<Edge>::iterator git, gend;
        for (git = edges.begin(), gend = edges.end(); git != gend; ++git)
        {
            Roaring64Map r;

            std::vector<uint64> cells = git->cells;
            std::vector<uint64>::iterator cit, cend;

            // printf("in the next round cell size: %lu\n", cells.size());
            for (cit = cells.begin(), cend = cells.end(); cit != cend; ++cit)
            {
                r.add((uint64_t)*cit);
            }

            // r.printf();

            //std::clock_t start;
            //double duration;

            //start = std::clock();

            // verify each trajectory by using bitmap intersection
            std::map<int, H_Index>::iterator vit, vend;
            for (vit = v_map.begin(), vend = v_map.end(); vit != vend; ++vit)
            {
                // if (vit == v_map.begin()) vit->second.cell_bitmap->printf();

                // TODO test, if intersect, count the number of candiate trajectories
                r &= *(vit->second.cell_bitmap);

                // if trajectory passing all cells, adding to the result
                // change the candidate H_Index
                if ( r.cardinality() == cells.size() )  
                {

                    uint64_t *ans = new uint64_t[r.cardinality()];
                    r.toUint64Array(ans);

                    unsigned long span = get_time_interval(ans, cells.size(), vit->second.s_map);

                    // printf("time span: %lu\n", span);

                    if (span < (T[1]-T[0]))
                    {
                        //std::cout << "cell id: " << ans[i] << std::endl;
                        t_map.insert( std::pair<int, std::vector<uint64> >(vit->first, cells) );

                        mid_map.insert(*vit);
                    }
                }
            }

            //duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;

            //std::cout<<"-------- verify cost: "<< duration <<'\n';
        }

        std::cout<<"-------- mid map size: "<< mid_map.size() <<'\n';

        // if the result keep balance or < cost_by_types, return
        if ( mid_map.size() == 0 || (v_map.size() - mid_map.size() < 10) ) return mid_map;
        else v_map = mid_map;
    }

    return v_map;
}

// TODO get neighborhood cells
// 1, 2, 3 types of POI intersection results( <tid, cell_id>, <1, 10002>, <3, 10003>), (<2, 20003>), (<1, 20001>, <2, 17293>), respectively
std::map<int, std::vector<unsigned long> >
Movix::cost_by_types(HMix *H, std::vector<std::vector<uint64> > type_cells, std::vector<unsigned long> times, std::vector<float> radius)
{
    // cell types are same as number of times
    //assert( type_cells.size() == times.size() );

    // get all bitmap indexes
    std::map<int, H_Index> last_map = H->get_hmap();
    std::map<int, H_Index>::iterator hit, hend;

    // std::vector<std::pair<int, unsigned long> > type_count_vec;
    std::map<int, std::vector<unsigned long> > type_count_map;
    std::vector<std::vector<uint64> >::iterator vit, vend;
    std::vector<unsigned long>::iterator tit = times.begin();

    for (vit = type_cells.begin(), vend = type_cells.end(); vit != vend; ++vit, ++tit)
    {
        // timestamp of one type POI
        unsigned long ts = *tit;

        // store candidate trajectory index
        std::map<int, H_Index> temp_map;
        std::vector<uint64> poi_vec = *vit;
        std::vector<uint64>::iterator pit, pend;

        // find in last bitmap indexes
        for (hit = last_map.begin(), hend = last_map.end(); hit != hend; ++hit)
        {

            for (pit = poi_vec.begin(), pend = poi_vec.end(); pit != pend; ++pit)
            {
                 // if trajectory passing by r
                if ( hit->second.cell_bitmap->contains((uint64_t)*pit) )  
                {
                    // verify timestamp on every trajectory, in 3 minutes
                    std::map<uint64, unsigned long> s_map = hit->second.s_map;
                    std::map<uint64, unsigned long>::iterator mit = s_map.find(*pit);

                    if (mit != hit->second.s_map.end())
                    {
                        unsigned long temp = mit->second;

                        std::map<int, std::vector<unsigned long> >::iterator cit = type_count_map.find(hit->first);
                        if (cit != type_count_map.end())
                            type_count_map[hit->first].push_back(temp);
                        else
                        {
                            std::vector<unsigned long> v;
                            v.push_back(temp);
                            type_count_map.insert( std::pair<int, std::vector<unsigned long> >(hit->first, v) );
                            
                        }

                        temp_map.insert( *hit );                           
                        break;  // only pass one cell of one type
                    }
                }
            }

        }

        // find in the reduced map in the next round
        last_map = temp_map;
    }

    printf("temp map size: %lu\n", type_count_map.size());

    return type_count_map;
}


template< int NUM_THREADS_PER_BLOCK >
__global__
void kernel_check_type(
        uint64          *   d_cells,                 // cells of trajectory
        uint64          *   d_poi_cells,             // cells of pois of one type
        Parameters          params,
        unsigned long   *   d_ts,                   // timestamps     
        unsigned long   *   times,                   // start and end time
        int             *   count                    // overlapped cells
    )
{
    // each thread represents one cell
    int tid = blockIdx.x * NUM_THREADS_PER_BLOCK + threadIdx.x;

    if (tid < params.num_cells)
    {
        uint64 cell_id = d_cells[tid];

        // check if cell id overlap a poi cell
        for (size_t k = 0; k < params.num_poi_cells; ++k)
        {
            // printf("---poi cell num: %lu, cell id: %llu\n", params.num_poi_cells, cell_id);
            if (cell_id == d_poi_cells[k] && d_ts[tid] > times[0] && d_ts[tid] < times[1])    
            {
                count[tid]++;
                break;
            }
        }
    }
}

template< int NUM_THREADS_PER_BLOCK >
__global__
void kernel_count_cell_types(int *d_count, int *d_tid, int *d_result, Parameters params)
{
    int thid = blockIdx.x * NUM_THREADS_PER_BLOCK + threadIdx.x;

    // if two consecutive tids are not equal, count all items with the same tid
    if (thid < params.num_cells - 1 && d_tid[thid] != d_tid[thid+1])
    { 
        size_t amount = 0;

        // printf("---%d, %d\n", d_tid[thid], d_tid[thid+1]);

        for (size_t k = thid+1; d_tid[k] == d_tid[k+1]; ++k)
        {   
            if (d_count[k] > 0) {
                amount += d_count[k];
                
            }
        }

        if (amount >= params.num_poi_types) d_result[thid] = d_tid[thid];
    }
}

thrust::device_vector<int>
Movix::cuda_cost_by_types(std::vector<std::vector<uint64> > type_cells, std::vector<unsigned long> times)
{
    std::vector<std::vector<uint64> >::iterator vit, vend;

    // suppose we have the max number of cells
    size_t num_cells = length;
    thrust::device_vector<int> d_count_vec(num_cells, 0);
    thrust::device_vector<unsigned long> d_times = times;
    thrust::device_vector<int> d_result_vec(num_cells, 0);

    const int NUM_THREADS_PER_BLOCK = 512;

    // for each type of pois, decide whether the bitmap of a trajectory overlaps a bitmap of pois
    for (vit = type_cells.begin(), vend = type_cells.end(); vit != vend; ++vit)
    {
        thrust::device_vector<uint64> d_poi_vec = type_cells[1];//*vit;

        // thrust::copy(d_poi_vec.begin(), d_poi_vec.end(), std::ostream_iterator<uint64>(std::cout, ","));

        size_t num_poi_cells = d_poi_vec.size();

        Parameters params( num_cells, num_poi_cells, type_cells.size());

        kernel_check_type<NUM_THREADS_PER_BLOCK><<<num_cells/NUM_THREADS_PER_BLOCK, NUM_THREADS_PER_BLOCK>>>(
            thrust::raw_pointer_cast(cell_vec.data()), 
            thrust::raw_pointer_cast(d_poi_vec.data()), 
            params,
            thrust::raw_pointer_cast(ts_vec.data()),
            thrust::raw_pointer_cast(d_times.data()),
            thrust::raw_pointer_cast(d_count_vec.data())
        );
    }

    // thrust::copy(d_count_vec.begin(), d_count_vec.end(), std::ostream_iterator<bool>(std::cout, ","));
    // count each cell of a trajectoy, only leaving trajectories with number of overlapping >= type number 
    Parameters params( num_cells, 0, type_cells.size() );
    kernel_count_cell_types<NUM_THREADS_PER_BLOCK><<<num_cells/NUM_THREADS_PER_BLOCK, NUM_THREADS_PER_BLOCK>>>( 
        thrust::raw_pointer_cast(d_count_vec.data()), 
        thrust::raw_pointer_cast(tid_vec.data()),  
        thrust::raw_pointer_cast(d_result_vec.data()),
        params
    );

    // thrust::copy(tid_vec.begin(), tid_vec.end(), std::ostream_iterator<int>(std::cout, ","));

    auto new_end = thrust::remove_if( d_result_vec.begin(), d_result_vec.end(), thrust::placeholders::_1 == 0);
    // thrust::copy(d_result_vec.begin(), new_end, std::ostream_iterator<int>(std::cout, ","));

    size_t len = new_end - d_result_vec.begin();
    thrust::device_vector<int> final_tid_vec(len);
    thrust::copy_n(d_result_vec.begin(), len, final_tid_vec.begin());

    return final_tid_vec;
}



std::vector<int> Movix::search_RTree(
        RTree *tree, 
        std::vector<uint64> cid_vec, 
        long s, long e)
{
    std::vector<int> points;

    for (int i = 0; i < cid_vec.size(); ++i)
    {
        RTree_Rect rect = {cid_vec[0]-10000000000ULL, cid_vec[0]+10000000000ULL, 
                                    (unsigned long)s, (unsigned long)e};
        printf("rect: %llu, %llu, %lu, %lu\n", rect.left, rect.right, rect.top, rect.bottom);

        int num = cpu_search(tree->root, &rect, points);
        
    }   

    return points;
}

/*std::vector<int> Movix::poi_search(HMix *H, POI_Data *D, std::vector<S2_POI> P, long s, long e)
{
    printf("size: %lu\n", P.size());
    assert(P.size() == 2);

    thrust::device_vector<int> d_tid_vec;

    // search in poi, return cell id
    for (int i = 0; i < P.size(); ++i)
    {
        std::vector<uint64> cid_vec = D->search(P[i].keywords, P[i].type);

        printf("---poi cell: %llu\n", cid_vec[0]);

        // search trajectory from R tree
        std::vector<int> points = search_RTree(tree, cid_vec, s, e);

        if (points.size() > 0)
        {
            d_tid_vec.resize(d_tid_vec.size() + points.size());
            thrust::copy(points.begin(), points.end(), d_tid_vec.end()-points.size());
        }
    }


    thrust::sort(d_tid_vec.begin(), d_tid_vec.end());
    auto new_end = thrust::unique(d_tid_vec.begin(), d_tid_vec.end());

    //thrust::copy(d_tid_vec.begin(), new_end, ostream_iterator<int>(std::cout, ","));
    
    std::vector<int> h_vec(new_end - d_tid_vec.begin());
    thrust::copy(d_tid_vec.begin(), new_end, h_vec.begin());
    return h_vec;
}
*/

std::vector<int> Movix::poi_search(HMix *H, HMix *LH, POI_Data *D, std::vector<S2_POI> P, std::vector<unsigned long> T, double v)
{
    // size_t number = 0;
    std::vector<std::vector<uint64> > cells_vec;
    
    // search in poi, return cell id
    for (int i = 0; i < P.size(); ++i)
    {
        std::vector<uint64> cid_vec = D->search(P[i].keywords, P[i].type);

        printf("---poi cell: %lu\n", cid_vec.size());

        if (!cid_vec.empty())
            cells_vec.push_back(cid_vec);
    }

    // std::clock_t start;
    // double duration;

    // start = std::clock();

    // // 1. estimate cost of different types of POI
    std::vector<float> r;
    std::map<int, std::vector<unsigned long> > type_count_map = cost_by_types(H, cells_vec, T, r);

    // duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
    // std::cout<<"poi type estimation cost: "<< duration <<'\n';

    // // request for the number of trajectories
    // std::map<int, uint64> r_map;
    // std::vector<std::pair<int, uint64> >::iterator iit, iend;
    // for (iit = type_count_vec.begin(), iend = type_count_vec.end(); iit != iend; iit++)
    // {
    //     r_map.insert(*iit);
    // }

    // // 2. estimate cost of POI in [s, e], according to distances of POIs
    // start = std::clock();

    std::sort(T.begin(), T.end());
    unsigned long s = T[0], e = T[T.size()-1];
    // unsigned long V[2] = {s, e};
    unsigned long span = e - s;

    // printf("s: %lu, e: %lu\n", s, e);
    // std::map<int, H_Index> time_count_vec = cost_by_time(H, cells_vec, V, v, r_map.size());

    // duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
    // std::cout<<"time constraints estimation cost: "<< duration <<'\n';

    // start = std::clock();

    // printf("time count size: %lu, type count size: %lu, after request: %lu\n", time_count_vec.size(), type_count_vec.size(), r_map.size());

    // // 3. compare and determine a query plan  
    // // std::sort(type_count_vec.begin(), type_count_vec.end(), 
    // //     [] (const std::pair<int, unsigned long>  & a, const std::pair<int, unsigned long> & b) -> bool
    // //     { return a.first < b.first;} );

    // std::map<int, H_Index> hmap = H->get_hmap();
    // std::map<int, H_Index> hmap_12 = LH->get_hmap();

    //  3.1 if dist_count < min(type_count_vec), execute the time interval constraint firstly
    // if (time_count_vec.size() < r_map.size()) 
    // {

    //     // 3.1.1 execute the time interval query
    //     // if a trajectory satisfy time constraints
    //     // std::map<int, std::vector<uint64> >::iterator tit, tend;
    //     // for (tit = time_count_vec.begin(), tend = time_count_vec.end(); tit != tend; ++tit)
    //     // {
    //     //     // get trajectory index by tid
    //     //     std::vector<uint64> cvec = tit->second;
            
    //     //     // std::cout <<  "----------cells id: " << c << std::endl;
    //     //     std::vector<uint64>::iterator cit, cend;
    //     //     for (cit = cvec.begin(), cend = cvec.end(); cit != cend; cit++)
    //     //     {
    //     //         S2CellId cell_id( *cit );

    //     //         for(S2CellId c = cell_id.child_begin(12); c != cell_id.child_end(12); c = c.next())
    //     //         {
    //     //             // if time in the h_index meets constraints
    //     //             std::map<int, H_Index>::iterator hit = hmap_12.find(tit->first);
    //     //             std::map<uint64, unsigned long> s_map = hit->second.s_map;
    //     //             std::map<uint64, unsigned long>::iterator mit = s_map.find( c.id() );

    //     //             if (mit != s_map.end())
    //     //             {
    //     //                 unsigned long ts = mit->second;
    //     //                 // printf("time in array: %lu\n", ts);
    //     //                 if (s < ts && ts < e) 
    //     //                 {
    //     //                     number++;
    //     //                     break;
    //     //                 }
    //     //             }
    //     //         }
    //     //     }

    //     //     // std::cout <<  "----------cells in level 12: " << i << std::endl;
    //     // }

    // // get all bitmap indexes
    // std::map<int, H_Index> last_map = time_count_vec;
    // std::map<int, H_Index>::iterator hit, hend;

    // // std::vector<std::pair<int, unsigned long> > type_count_vec;
    // std::vector<std::vector<uint64> >::iterator vit, vend;
    // std::vector<unsigned long>::iterator tit = T.begin();

    // for (vit = cells_vec.begin(), vend = cells_vec.end(); vit != vend; ++vit, ++tit)
    // {
    //     // timestamp of one type POI
    //     unsigned long ts = *tit;

    //     // store candidate trajectory index
    //     std::map<int, H_Index> temp_map;
    //     std::vector<uint64> poi_vec = *vit;
    //     std::vector<uint64>::iterator pit, pend;
    //     for (pit = poi_vec.begin(), pend = poi_vec.end(); pit != pend; ++pit)
    //     {
    //         // find in last bitmap indexes
    //         for (hit = last_map.begin(), hend = last_map.end(); hit != hend; ++hit)
    //         {

    //             S2CellId cell_id( *pit );

    //             // 3.1.1 execute the time interval query
    //             // if a trajectory satisfy time constraints
    //             for(S2CellId c = cell_id.child_begin(12); c != cell_id.child_end(12); c = c.next())
    //             {

    //                 std::map<int, H_Index>::iterator hit_12 = hmap_12.find(hit->first);

    //                 std::map<uint64, unsigned long> s_map = hit_12->second.s_map;
    //                 std::map<uint64, unsigned long>::iterator mit = s_map.find( c.id() );

    //                 if (mit != s_map.end())
    //                 {
    //                     unsigned long ts = mit->second;
    //                     // printf("time in array: %lu\n", ts);
    //                     if (s < ts && ts < e) 
    //                     {
    //                         temp_map.insert( *hit );
    //                         break;
    //                     }
    //                 }
    //             }

    //             //  // if trajectory passing by r
    //             // if ( hit_12->second.cell_bitmap->contains((uint64_t)*pit) )  
    //             // {
    //             //     // verify timestamp on every trajectory, in 3 minutes
    //             //     std::map<uint64, unsigned long> s_map = hit->second.s_map;
    //             //     std::map<uint64, unsigned long>::iterator mit = s_map.find(*pit);

    //             //     if (mit != hit->second.s_map.end())
    //             //     {
    //             //         unsigned long temp = mit->second;
    //             //         if (abs(ts - temp) < 3 * 60 * 1000) // in 3 minutes  
    //             //         {
    //             //             type_count_vec.push_back( std::pair<int, unsigned long>(hit->first, ts) );
    //             //             temp_map.insert( *hit );
    //             //         }
    //             //     }
    //             //     //type_count++;
    //             // }
    //         }
    //     }

    //     // find in the reduced map in the next round
    //     last_map = temp_map;
    // }

    //     return last_map.size();

    // }
    // //  3.2 else, execute the query of multiple types of POI constraints
    // else
    // {
    //     printf("into second round-------------\n");

    //     // 3.2.1 
    //     // Trie *p_t = new Trie(v, (T[T.size()-1]-T[0]));

    //     std::vector<Measure> L;
    //     std::vector<Pair> R;
        std::vector<int> candidates;

    //     // for each trajectory, find the minimum timestamp and the maximum timestamp in the type_count_vec
        std::map<int, std::vector<unsigned long> >::iterator tit, tend;

        // std::sort(type_count_vec.begin(), type_count_vec.end(),
        //      [] (const std::pair<int, unsigned long> & a, const std::pair<int, unsigned long> & b) -> bool {
        //         if (a.first < b.first) return true;
        //         return false;
        //      } );

    //     // size_t idx = 0;
        // unsigned long ts_min = -1, ts_max = 0;
        // int last_tid = 0;
        for (tit = type_count_map.begin(), tend = type_count_map.end(); tit != tend; ++tit)
        {
            std::vector<unsigned long> time_vec = tit->second;
            std::sort(time_vec.begin(), time_vec.end());

            size_t len = time_vec.size();
            assert( len > 0);
            if (time_vec[len-1] - time_vec[0] < span)
            {
                std::vector<int>::iterator cit = std::find(candidates.begin(), candidates.end(), tit->first);
                if (cit == candidates.end())
                    candidates.push_back(tit->first);
            }
        }
            // // find the first timestamp and the last timestamp of the trajectory
            // if (tit->first == last_tid)
            // {
            //     if (ts_max < tit->second) ts_max = tit->second;
            //     if (ts_min > tit->second) ts_min = tit->second; 
            // }
            // else
            // {
            //     // if the time interval is less than the time constraint
            //     if (ts_max > 0 && ts_max - ts_min < span ) 
            //     {
            //         printf("max time: %lu, min time: %lu, span: %lu\n", ts_max, ts_min, span);

            //         std::vector<int>::iterator cit = std::find(candidates.begin(), candidates.end(), tit->first);
            //         if (cit == candidates.end())
            //             candidates.push_back(tit->first);
            //     }

                
            // }

            // last_tid = tit->first;
    // }

    //     std::vector<int>::iterator cit, cend;
    //     for (cit = candidates.begin(), cend = candidates.end(); cit != cend; ++cit)
    //     {
    //         std::map<int, H_Index>::iterator hit_12 = hmap_12.find(tit->first);
    //     }


            // if (tit->first == last_tid)
            // {
            //     S2CellId cell_id( tit->second );

            //     // 3.1.1 execute the time interval query
            //     // if a trajectory satisfy time constraints
            
            //     for(S2CellId c = cell_id.child_begin(12); c != cell_id.child_end(12); c = c.next())
            //     {

            //         std::map<int, H_Index>::iterator hit_12 = hmap_12.find(tit->first);

            //         std::map<uint64, unsigned long> s_map = hit_12->second.s_map;
            //         std::map<uint64, unsigned long>::iterator mit = s_map.find( c.id() );

            //         if (mit != s_map.end())
            //         {
            //             unsigned long ts = mit->second;
            //             // printf("time in array: %lu\n", ts);
            //             if (s < ts && ts < e) 
            //             {
            //                 if (ts_max < ts) ts_max = ts;
            //                 if (ts_min > ts) ts_min = ts; 
            //             }
            //         }
            //     }
            // }
            // else
            // {
            //     // if the time interval is less than the time constraint
            //     if (ts_max > 0 && ts_max - ts_min < span ) number++;
            // }

            // last_tid = tit->first;

            // for (mit = tmap.begin(), mend = tmap.end(); mit != mend; ++mit)
            // {
            //     // if timestmaps in all types meets constraints
            //     std::map<int, H_Index>::iterator hit = hmap.find(mit->first);
            //     unsigned long *s_array = hit->second.s_array;
            //     uint64 cell_id = mit->second;
            //     unsigned long ts =  s_array[highBytes(cell_id)];

            //     // initialize measure
            //     if (idx == 0)
            //     {
            //         Measure mes;
            //         mes.interval = 0;
            //         mes.cells.push_back(mit->second);
            //         mes.latest = ts;
            //         L.push_back(mes);
            //     }
            //     else 
            //     {
            //         Pair p;
            //         p.cell_id = mit->second;
            //         p.time = ts;
            //         R.push_back(p);
            //     }
            // }

            // ++idx;
            
            // if (!R.empty())
            //     L = p_t->multix(L, R);
        // }

        // number = L.size();
    // }

    return candidates;
}

