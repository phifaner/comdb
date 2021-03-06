#include <sys/mman.h>
#include <fcntl.h>
#include <time.h>
#include <unistd.h>
#include <sys/stat.h>
#include <dirent.h>

#include <thrust/count.h>
#include <thrust/copy.h>
#include <thrust/sequence.h>

#include "hash_table.h"
#include "parser.h"


/* !!!  id list includes translated unsigned long value from (trajectory id, date) */
int DataLoader::load_data_by_ids(const unsigned long * id_list, int len, comdb *db)
{
    for (int i = 0; i < len; i++)
    {
        // translate id into file path
        const unsigned long id = id_list[i];
        struct item * t = get_by_key(id);
        const char * path = t->value;

        printf("-----%lu, %s\n", t->key, path);

        // load data from file 
        load_data_wz(path, db);        
    }

    return 1;
}

int DataLoader::load_data_wz(const char * filename, comdb * db)
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
                    
    // the following line is very important, to get the line starting position in the data
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
    thrust::device_vector<long> _time_vec(cnt);
    thrust::device_vector<int> _state_vec(cnt);
    thrust::device_vector<int> _speed_vec(cnt);
    thrust::device_vector<int> _dir_vec(cnt);

    db->col_id_vec.resize(db->col_id_vec.size() + cnt);
    db->col_lat_vec.resize(db->col_lat_vec.size() + cnt);
    db->col_lon_vec.resize(db->col_lon_vec.size() + cnt);
    db->col_time_vec.resize(db->col_time_vec.size() + cnt);
    db->col_state_vec.resize(db->col_state_vec.size() + cnt);
    db->col_speed_vec.resize(db->col_speed_vec.size() + cnt);
    db->col_dir_vec.resize(db->col_dir_vec.size() + cnt);

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

    // parse time column
    index_cnt[0] = 23;
    gpu_date date_ff((const char*)thrust::raw_pointer_cast(time_vec.data()),
             (long int*)thrust::raw_pointer_cast(_time_vec.data()),
                    thrust::raw_pointer_cast(index_cnt.data()));
    thrust::for_each(begin, begin+cnt, date_ff);

    // parse state column
    index_cnt[0] = 1;
    gpu_atoi atoi_ff_state((const char*)thrust::raw_pointer_cast(state_vec.data()),
            (int*)thrust::raw_pointer_cast(_state_vec.data()),
                thrust::raw_pointer_cast(index_cnt.data()));
    thrust::for_each(begin, begin+cnt, atoi_ff_state);

    // parse speed column
    index_cnt[0] = 2;
    gpu_atoi atoi_ff_speed((const char*)thrust::raw_pointer_cast(speed_vec.data()),
            (int*)thrust::raw_pointer_cast(_speed_vec.data()),
                thrust::raw_pointer_cast(index_cnt.data())); 
    thrust::for_each(begin, begin+cnt, atoi_ff_speed);

    // parse direction column
    index_cnt[0] = 1;
    gpu_atoi atoi_ff_dir((const char*)thrust::raw_pointer_cast(direction_vec.data()),
            (int*)thrust::raw_pointer_cast(_dir_vec.data()),
                thrust::raw_pointer_cast(index_cnt.data())); 
    thrust::for_each(begin, begin+cnt, atoi_ff_dir);

    // add new column values in containers
    thrust::copy(_id_vec.begin(), _id_vec.end(), db->col_id_vec.end()-cnt) ; 
    thrust::copy(_lat_vec.begin(), _lat_vec.end(), db->col_lat_vec.end()-cnt);
    thrust::copy(_lon_vec.begin(), _lon_vec.end(), db->col_lon_vec.end()-cnt);
    thrust::copy(_time_vec.begin(), _time_vec.end(), db->col_time_vec.end()-cnt);
    thrust::copy(_state_vec.begin(), _state_vec.end(), db->col_state_vec.end()-cnt);
	thrust::copy(_speed_vec.begin(), _speed_vec.end(), db->col_speed_vec.end()-cnt);
    thrust::copy(_dir_vec.begin(), _dir_vec.end(), db->col_dir_vec.end()-cnt);    
    
    db->size = db->col_id_vec.size();
 
   // std::cout << "db size: " << db->size << std::endl;   
    //thrust::copy(db->col_lat_vec.begin(), db->col_lat_vec.end(), std::ostream_iterator<double>(std::cout, ","));
     
    //thrust::copy(_id_vec.begin(), _id_vec.end(), std::ostream_iterator<int>(std::cout, ","));
    //thrust::copy(id_vec.begin(), id_vec.end(), std::ostream_iterator<char>(std::cout, ","));

    return 0;
}

/* this function only used for load taxi data of beijing */
int DataLoader::load_data_bj(const char* filename, comdb &db)
{
    // get file size and prepare to map file
    FILE* f = fopen(filename, "r");
    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    thrust::device_vector<char> dev(file_size);
    fclose(f);

    // map file from disk to memory
    char* p_mmap;

    int fd = open(filename, O_RDONLY);
    if (fd == -1)
    {
        perror("open");
        return EXIT_FAILURE;
    }

    // printf("file size: %ld, file id: %d, file name: %s\n", file_size, fd, filename);

   /* if (file_size == 0) 
    {
    perror("file size");
    //printf("%s file size 0 \n", filename);
    return EXIT_FAILURE;
    }
*/
    if (file_size > 0)
        p_mmap = (char*) mmap(0, file_size, PROT_READ, MAP_SHARED, fd, 0);
    
    if (p_mmap == MAP_FAILED)
    {
        perror("mmap");
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
    //std::cout << "There are " << cnt << " total points " << std::endl;

    // find the position of '\n' to locate each line
    thrust::device_vector<int> line_index(cnt+1);
    line_index[0] = -1;
    
    // the following line is very important, to get the line starting position in the data
    thrust::copy_if(thrust::make_counting_iterator((unsigned int)0),
            thrust::make_counting_iterator((unsigned int) file_size),
            dev.begin(), line_index.begin()+1, is_break());

    // initialize column vectors
    thrust::device_vector<char> id_vec(cnt*6, 0);
    //thrust::fill(id_vec.begin(), id_vec.end(), 0);
    thrust::device_vector<char> time_vec(cnt*19, 0);
    //thrust::fill(time_vec.begin(), time_vec.end(), 0);
    thrust::device_vector<char> longitude_vec(cnt*9, 0);
    //thrust::fill(longitude_vec.begin(), longitude_vec.end(), 0);
    thrust::device_vector<char> latitude_vec(cnt*8, 0);
    //thrust::fill(latitude_vec.begin(), latitude_vec.end(), 0);

    thrust::device_vector<char*> dest(4);
    dest[0] = thrust::raw_pointer_cast(id_vec.data());
    dest[1] = thrust::raw_pointer_cast(time_vec.data());
    dest[2] = thrust::raw_pointer_cast(longitude_vec.data());
    dest[3] = thrust::raw_pointer_cast(latitude_vec.data()); 
 
    // set field max length
    thrust::device_vector<unsigned int> dest_len(4);
    dest_len[0] = 6;
    dest_len[1] = 19;
    dest_len[2] = 9;
    dest_len[3] = 8;

    // set index for each column, default 4 columns
    thrust::device_vector<unsigned int> index(4);
    thrust::sequence(index.begin(), index.end());
 
    // set fields count
    thrust::device_vector<unsigned int> index_cnt(1);
    index_cnt[0] = 4;

    thrust::device_vector<char> sep(1);
    sep[0] = ',';

    thrust::counting_iterator<unsigned int> begin(0);
    parser parse((const char*)thrust::raw_pointer_cast(dev.data()), (char**)thrust::raw_pointer_cast(dest.data()),
            thrust::raw_pointer_cast(index.data()), thrust::raw_pointer_cast(index_cnt.data()),
            thrust::raw_pointer_cast(sep.data()), thrust::raw_pointer_cast(line_index.data()),
        thrust::raw_pointer_cast(dest_len.data())
            );
    thrust::for_each(begin, begin+cnt, parse);

    /*std::cout << cnt << "------" << std::endl;
    for (int i = 0; i < cnt * 4; )
    {
        //for (int j = 0; j < 8; j++)
        //{
            std::cout << longitude_vec[i++];
        //} 
    }
    std::cout << std::endl;*/

    // initialize comdb column containers
    thrust::device_vector<int> _id_vec(cnt);   
    thrust::device_vector<double> _lat_vec(cnt);
    thrust::device_vector<double> _lon_vec(cnt);
    thrust::device_vector<long> _time_vec(cnt);
    
    try {
        db.col_id_vec.resize(db.col_id_vec.size() + cnt);
    db.col_lat_vec.resize(db.col_lat_vec.size() + cnt);
        db.col_lon_vec.resize(db.col_lon_vec.size() + cnt);
        db.col_time_vec.resize(db.col_time_vec.size() + cnt);
    }
    catch (std::length_error)
    {
    std::cout << "--------resize exceed max length! ---------" << std::endl;
    }
    
    //res_lat_vec.resize(res_lat_vec.size() + cnt);

    // parse id column
    index_cnt[0] = 6;
    gpu_atoi atoi_ff((const char*) thrust::raw_pointer_cast(id_vec.data()), 
            (int*)thrust::raw_pointer_cast(_id_vec.data()),
                thrust::raw_pointer_cast(index_cnt.data()));
    thrust::for_each(begin, begin+cnt, atoi_ff);


    // parse latitude column
    index_cnt[0] = 8;
    gpu_atof atof_ff_lat((const char*)thrust::raw_pointer_cast(latitude_vec.data()), 
            (double*)thrust::raw_pointer_cast(_lat_vec.data()),
                thrust::raw_pointer_cast(index_cnt.data()));
    thrust::for_each(begin, begin+cnt, atof_ff_lat);

    // parse longitude column
    index_cnt[0] = 9;
    gpu_atof atof_ff_lon((const char*)thrust::raw_pointer_cast(longitude_vec.data()),
            (double*)thrust::raw_pointer_cast(_lon_vec.data()),
                thrust::raw_pointer_cast(index_cnt.data()));
    thrust::for_each(begin, begin+cnt, atof_ff_lon);

    // parse time column
    index_cnt[0] = 19;
    gpu_date date_ff((const char*)thrust::raw_pointer_cast(time_vec.data()),
            (long int*)thrust::raw_pointer_cast(_time_vec.data()),
                thrust::raw_pointer_cast(index_cnt.data()));
    thrust::for_each(begin, begin+cnt, date_ff);

    // add new column values in containers
    thrust::copy(_id_vec.begin(), _id_vec.end(), db.col_id_vec.end()-cnt) ; 
    thrust::copy(_lat_vec.begin(), _lat_vec.end(), db.col_lat_vec.end()-cnt);
    thrust::copy(_lon_vec.begin(), _lon_vec.end(), db.col_lon_vec.end()-cnt);
    thrust::copy(_time_vec.begin(), _time_vec.end(), db.col_time_vec.end()-cnt);

    db.size = db.col_id_vec.size();
    // print id column 
    //thrust::copy(db.col_lat_vec.begin(), db.col_lat_vec.end(), std::ostream_iterator<double>(std::cout, ","));
    // thrust::copy(_id_vec.begin(), _id_vec.end(), std::ostream_iterator<int>(std::cout, ","));

    return 0;
}

std::vector<char*> DataLoader::find_files( const char* folder )
{
    DIR * dir;
    struct dirent * ent;

    std::vector<char*> file_vec;

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
                
                find_files( path);
             }
             else
             {
                char *file_name = new char[100];
                sprintf( file_name, "%s/%s", folder, ent->d_name );
                file_vec.push_back(file_name);
                //printf("file number: %d, %s\n", num_file, file_name);
             }
        }
        
        closedir( dir );
    }
    else
    {
        perror( "read file error!" );
    }

    return file_vec;
} 
