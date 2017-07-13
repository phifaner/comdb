#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <fcntl.h>
#include <dirent.h>
#include <math.h>

#include <thrust/device_vector.h>
#include <thrust/count.h>
#include <thrust/copy.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

#include "parser.h"
#include "rtree.h"

thrust::device_vector<int>      index_array;
thrust::device_vector<float4>   rect_array;

char ** file_array;
int num_file;

void init_file_array()
{
    num_file = 0;

    file_array = (char**) malloc(sizeof(char*) * 10000);    
    
    for (int i = 0; i < 10000; ++i)
        file_array[i] = (char*)malloc(sizeof(char)*100);
}

int is_dir(const char * path)
{
    struct stat statbuf;
    if (stat(path, &statbuf) !=0) return 0;

    return S_ISDIR(statbuf.st_mode);
}

void list_dir(const char * folder, int level)
{
    DIR * dir;
    struct dirent * ent;

    if ((dir = opendir(folder)) != NULL)
    {
        while ((ent = readdir(dir)) != NULL)
        {
            if (ent->d_type == DT_DIR) 
            {
                char path[100];
                int len = snprintf(path, sizeof(path)-1, "%s/%s", folder, ent->d_name);
                path[len]  = 0;

                // skip self and parent
                if (strcmp(ent->d_name, ".") == 0 || strcmp(ent->d_name, "..") == 0)
                    continue;

                if (strlen(folder)+strlen(ent->d_name)+2 > sizeof(path))
                    fprintf(stderr, "name %s %s too long\n", folder, ent->d_name);
            
                //printf("%*s[%s]\n", level*2, "", ent->d_name);
               
                list_dir(path, level+1);
            }
            else
            { 
                char file_name[100];
                //printf("%*s- %s\n", level*2, "", ent->d_name);
                
                sprintf(file_name, "%s/%s", folder, ent->d_name);

                strcpy(file_array[num_file++], file_name);
            }
        }

        closedir(dir);
    }
    else
    {
        perror("read file error!");
        return;    
    }

    //printf("num_file--------%d\n", num_file);

}

void traj_bbx(thrust::device_vector<float4> & d_bbox_vec)
{
    for (int i = 0; i < num_file; i++) 
    {
        float4 bbox;
        compute_bbx(file_array[i], bbox);
        d_bbox_vec.push_back(bbox);
     
        //printf("%lu----%s\n", d_bbox_vec.size(), file_array[i]);
    }

    free(file_array);

    //thrust::for_each(d_bbox_vec.begin(), d_bbox_vec.end(), [=] __device__ (float4 f) {printf("%f\n", f.x);});   
}

int compute_bbx(char * filename, float4 &bbox)
{
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
    thrust::device_vector<char> id_vec(cnt*4, 0);
    thrust::device_vector<char> longitude_vec(cnt*9);
    thrust::device_vector<char> latitude_vec(cnt*8);
    thrust::device_vector<char> time_vec(cnt*23, 0);
    thrust::device_vector<char> state_vec(cnt*1, 0);
    thrust::device_vector<char> speed_vec(cnt*2, 0);
    thrust::device_vector<char> direction_vec(cnt*1, 0);

    thrust::device_vector<char*> dest(7);
    dest[0] = thrust::raw_pointer_cast(id_vec.data());
    dest[1] = thrust::raw_pointer_cast(longitude_vec.data());
    dest[2] = thrust::raw_pointer_cast(latitude_vec.data());
    dest[3] = thrust::raw_pointer_cast(time_vec.data());
    dest[4] = thrust::raw_pointer_cast(state_vec.data());
    dest[5] = thrust::raw_pointer_cast(speed_vec.data());
    dest[6] = thrust::raw_pointer_cast(direction_vec.data());

    // set field max length
    thrust::device_vector<unsigned int> dest_len(7);
    dest_len[0] = 4;
    dest_len[1] = 9;
    dest_len[2] = 8;
    dest_len[3] = 23;
    dest_len[4] = 1;
    dest_len[5] = 2;
    dest_len[6] = 1;

    // set index for each column, default 7 columns
    thrust::device_vector<unsigned int> index(7);
    thrust::sequence(index.begin(), index.end());

    // set fields count
    thrust::device_vector<unsigned int> index_cnt(1);
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
    thrust::for_each(begin, begin+cnt, parse);

    // initialize database column container
    thrust::device_vector<int> _id_vec(cnt);
    thrust::device_vector<double> _lon_vec(cnt);
    thrust::device_vector<double> _lat_vec(cnt);

    // parse id column
    index_cnt[0] = 4;
    gpu_atoi atoi_id((const char*)thrust::raw_pointer_cast(id_vec.data()),
            (int*)thrust::raw_pointer_cast(_id_vec.data()),
            thrust::raw_pointer_cast(index_cnt.data()));
    thrust::for_each(begin, begin+cnt, atoi_id);

    // parse longitude column
    index_cnt[0] = 9;
    gpu_atof atof_ff_lon((const char*)thrust::raw_pointer_cast(longitude_vec.data()),
            (double*)thrust::raw_pointer_cast(_lon_vec.data()),
            thrust::raw_pointer_cast(index_cnt.data()));
    thrust::for_each(begin, begin+cnt, atof_ff_lon);

    // parse latitude column
    index_cnt[0] = 8;
    gpu_atof atof_ff_lat((const char*)thrust::raw_pointer_cast(latitude_vec.data()),
            (double*)thrust::raw_pointer_cast(_lat_vec.data()),
            thrust::raw_pointer_cast(index_cnt.data()));
    thrust::for_each(begin, begin+cnt, atof_ff_lat);

    // order coordinates to get bounding box
    thrust::sort_by_key(_lon_vec.begin(), _lon_vec.end(), _lat_vec.begin());

    //thrust::copy(_lon_vec.begin(), _lon_vec.end(), std::ostream_iterator<double>(std::cout, ","));

    bbox =  make_float4(_lon_vec[0], _lat_vec[0], _lon_vec[cnt-1], _lat_vec[cnt-1]);
    
    free(filename);

    return 1;
}

void init(int fanout)
{
    size_t depth = log(num_file)/log(fanout) + 1;
    size_t node_num = pow(fanout, depth);
    index_array.resize(node_num);
    rect_array.resize(node_num);
}

void sort_lowx(thrust::device_vector<float4> &rect_vec)
        //thrust::device_vector<double> &lon_vec_low, 
        //thrust::device_vector<double> &lat_vec_low,
        //thrust::device_vector<double> &lon_vec_upper,
        //thrust::device_vector<double> &lat_vec_upper)
{
    //typedef thrust::device_vector<double>::iterator DoubleIterator;

    //thrust::tuple<DoubleIterator, DoubleIterator, DoubleIterator>
    //    coord_tuple = thrust::make_tuple(lat_vec_low.begin(),
    //                   lon_vec_upper.begin(),
    //                   lat_vec_upper.begin());

    //thrust::sort_by_key(lon_vec_low.begin(), lon_vec_low.end(), coord_tuple);
       
}

void pack_level(
        thrust::device_vector<float4> in_rect_vec,
        //thrust::device_vector<double> lon_min_vec,
        //thrust::device_vector<double> lat_min_vec,
        //thrust::device_vector<double> lon_max_vec,
        //thrust::device_vector<double> lat_max_vec,
        int fanout, int num_level,
        thrust::device_vector<int> & index_vec,
        thrust::device_vector<float4> & out_rect_vec
        )
{
   // thrust::device_vector<double> out_lon_min_vec,
    //                              out_lat_min_vec,
      //                            out_lon_max_vec,
      //                            out_lat_max_vec;
    
    //thrust::device_vector<float4> out_rect_vec(num_level);

    printf("=========before pack %lu\n", index_vec.size());

    /* given an ordered series of node items, determine their parent nodes */   
    pack_kernel pack(
            thrust::raw_pointer_cast(in_rect_vec.data()),
            //thrust::raw_pointer_cast(lon_min_vec.data()),
            //thrust::raw_pointer_cast(lat_min_vec.data()),
            //thrust::raw_pointer_cast(lon_max_vec.data()),
            //thrust::raw_pointer_cast(lat_max_vec.data()),
            fanout,
            thrust::raw_pointer_cast(index_vec.data())
            //thrust::raw_pointer_cast(out_rect_vec.data())
            //thrust::raw_pointer_cast(out_lon_min_vec.data()),
            //thrust::raw_pointer_cast(lat_min_vec.data()),
            //thrust::raw_pointer_cast(lon_max_vec.data()),
            //thrust::raw_pointer_cast(lat_max_vec.data())
        );

    thrust::for_each(
            thrust::make_counting_iterator((unsigned int) 0),
            thrust::make_counting_iterator((unsigned int) index_vec.size()),
            pack);       
}

void bulk_load(int fanout, thrust::device_vector<float4> & bbx)
{
    //thrust::device_vector<float4> in_rect_vec(leaf_item_num);
    //thrust::device_vector<double> lon_min_vec(leaf_node_num);
    //thrust::device_vector<double> lat_min_vec(leaf_node_num);
    //thrust::device_vector<double> lon_max_vec(leaf_node_num);
    //thrust::device_vector<double> lat_max_vec(leaf_node_num);

    /* copy rectangle from CPU to GPU mem */
   // for (size_t i = 0; i < leaf_item_num; i++)
   // {
     //   in_rect_vec[i] = bbx[i];
            //make_float4(bbx[i].x_min,  bbx[i].y_min, 
             //               bbx[i].x_max, bbx[i].y_max);
    //}

    /* sort data rectangles on ascending lon-coord of left-bottom corner */
    //sort_lowx(in_rect_vec);
    thrust::sort(bbx.begin(), bbx.end(), float4_greater());

    printf("bbox size: %lu\n", bbx.size());
    //thrust::copy(bbx.begin(), bbx.end(), os_stream::iterator<double>(std::cout, ","));

    int l = log(num_file)/log(fanout) + 1;
    
    printf("level : %d\n", l);
    
    /* pack items level by level from bottom to up */
    while (l--)
    {
        size_t num_level = pow(fanout, l);
        thrust::device_vector<float4> out_rect_vec(num_level);
        //thrust::device_vector<double> out_lon_min_vec(num_level);
        //thrust::device_vector<double> out_lat_min_vec(num_level);
        //thrust::device_vector<double> out_lon_max_vec(num_level);
        //thrust::device_vector<double> out_lat_max_vec(num_level);

        thrust::device_vector<int> index_vec(num_level);
        pack_level(
                bbx,
                fanout, num_level, 
                index_vec,
                out_rect_vec);
                //thrust::raw_pointer_cast(lon_min_vec.data()),
                //thrust::raw_pointer_cast(lat_min_vec.data()),
                //thrust::raw_pointer_cast(lon_max_vec.data()),
                //thrust::raw_pointer_cast(lat_max_vec.data()),
                //fanout,
                //thrust::raw_pointer_cast(index_vec.data()),
                //thrust::raw_pointer_cast(out_lon_min_vec.data()),
                //thrust::raw_pointer_cast(out_lat_min_vec.data()),
                //thrust::raw_pointer_cast(out_lon_max_vec.data()),
                //thrust::raw_pointer_cast(out_lat_max_vec.data())
                //thrust::raw_pointer_cast(out_rect_vec.data())
        // );

        //printf("============\n"); 
        /* put index and bounding box in global array */
        size_t first = pow(fanout, l-1) - 1;    // first position in rect array
        //thrust::copy_n(index_vec.begin(), num_level, index_array.begin()+first);

        //thrust::copy(index_vec.begin(), index_vec.end(), std::ostream_iterator<int>(std::cout));

        //thrust::copy_n(out_rect_vec.begin(), num_level, rect_array.begin()+first);

    } 

    /*void compute_height(int K, int B)
    {
        int height = ceiling_log(K, B);
        int total_node = 0;

        // used for number of nodes on each level
        int * nnarry = malloc(sizeof(int) * height);
        
        // we already know the number of nodes on the last level
        nnarry[height-1] = ceiling(K/B);
        total_node =  nnarry[height-1];
        for (int i = height-1; i > 0; i--)
        {
            nnarry[i] = ceiling(nnarry[i+1]/B);
            total_node += nnarry[i];
        }
    
        // used for index of each node
        int *  =  malloc(sizeof(int) * total_node * B);
        for (int j = 0; j < height-1; j++)
        {
            for (int i = 0; i < nnarry[j]*B; i++)
            {
                
            }   
        }
  }  */

}

int node_level(int level, int total_level, int * larray)
{
    int num = 0;
    for (int i = level+1; i <= total_level; i++)
    {
        num += larray[i];           // node number of each level
    }

    return num;
}

int parent_item_index(int level, int depth, int node_order, int B, int * larray)
{
    // parent node order of level
    int parent_node = floor((double)node_order/B);
    // item number of parent node
    int item_num = node_order%B;

    // compute the item order in the tree
    int item_order = node_level(level+1, depth, larray);
    item_order += parent_node;          // node order of the parent level
    item_order *= B;                    // each node has B items
    item_order += item_num;             // item number in a node
                                            
    return item_order;
}


void compute_index(int K, int B)
{
    // for number of each level
    int * larray = (int *)malloc(sizeof(int) * 20);
                            
    // total number of nodes
    int total_num = 0;

    // compute number of each level, bottom-up
    int level = 0;
    int level_num = K;

    larray[level] = level_num;
    total_num = level_num;

    while (level_num != 1) {
        level_num = (int) ceil((double)level_num/B);
        //printf("level num %d, level : %d\n", level_num, level);
        larray[++level] = level_num;
        total_num += level_num;
    }

    // for index of each item
    int * index = (int*) calloc(total_num*B, sizeof(int));

    // given level and node order, compute item to index the node
    for (int i = 0; i < level; i++)
    {
        for (int j = 0; j < larray[i]; j++)
        {
            // node index
            int inode = node_level(i, level, larray);
            inode += j;
            
            // leaf node index = -j
            if (i == 0) 
            {
                for (int m = 0; m < B; m++)
                    index[inode*B+m] = -1 * (j * B + m + 1);
            }
            
            // insert current node into parent node
            int item = parent_item_index(i, level, j, B, larray);
            index[item] = inode * B;
            
            //printf("item: %d, index: %d\n", item, index[item]);
        }
    }

    for (int i = 0; i < total_num*B; i++)
        printf("item: %d, index: %d\n", i, index[i]);

                                                                                            
}
