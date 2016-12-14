#include <thrust/for_each.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/remove.h>

#include "comdb.h"
#include "parser.h"

#include <sys/mman.h>
#include <unistd.h>
#include <fcntl.h>
#include <time.h>

/*extern thrust::device_vector<int> col_id_vec;			// object id
	
extern thrust::device_vector<double> col_lat_vec;		// latitude

extern thrust::device_vector<double> col_lon_vec;		// longitude

extern thrust::device_vector<long int> col_time_vec;		// time

extern thrust::device_vector<double> res_lat_vec;		// query result of latitude

extern thrust::device_vector<int> res_id_vec;			// query result of id
*/
int load_data(const char* filename, comdb &db)
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

	//printf("file size: %ld, file id: %d, file name: %s\n", file_size, fd, filename);

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
    gpu_atoi atoi_ff((const char*) thrust::raw_pointer_cast(id_vec.data()), (int*)thrust::raw_pointer_cast(_id_vec.data()),
			thrust::raw_pointer_cast(index_cnt.data()));
    thrust::for_each(begin, begin+cnt, atoi_ff);


    // parse latitude column
    index_cnt[0] = 8;
    gpu_atof atof_ff_lat((const char*)thrust::raw_pointer_cast(latitude_vec.data()), (double*)thrust::raw_pointer_cast(_lat_vec.data()),
			thrust::raw_pointer_cast(index_cnt.data()));
    thrust::for_each(begin, begin+cnt, atof_ff_lat);

    // parse longitude column
    index_cnt[0] = 9;
    gpu_atof atof_ff_lon((const char*)thrust::raw_pointer_cast(longitude_vec.data()), (double*)thrust::raw_pointer_cast(_lon_vec.data()),
			thrust::raw_pointer_cast(index_cnt.data()));
    thrust::for_each(begin, begin+cnt, atof_ff_lon);

    // parse time column
    index_cnt[0] = 19;
    gpu_date date_ff((const char*)thrust::raw_pointer_cast(time_vec.data()), (long int*)thrust::raw_pointer_cast(_time_vec.data()),
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
    //thrust::copy(_id_vec.begin(), _id_vec.end(), std::ostream_iterator<int>(std::cout, ","));

    return 0;
}

void comdb::select_by_id(int id)
{
    id_equal ieq(id);
   
    size_t _size = col_id_vec.size(); 	
    res_lat_vec.resize(_size);
    res_id_vec.resize(_size);
    res_lon_vec.resize(_size);
    res_time_vec.resize(_size);

        //std::cout << "____-------__________________"  << std::endl; 	

    thrust::for_each(
            thrust::make_zip_iterator(thrust::make_tuple(col_id_vec.begin(), col_lat_vec.begin(),
		col_lon_vec.begin(), col_time_vec.begin(),  res_id_vec.begin(), res_lat_vec.begin(), res_lon_vec.begin(), res_time_vec.begin())),
            thrust::make_zip_iterator(thrust::make_tuple(col_id_vec.end(), col_lat_vec.end(), 
		col_lon_vec.end(), col_time_vec.end(), res_id_vec.end(), res_lat_vec.end(), res_lon_vec.end(), res_time_vec.end())),
            ieq);

 //	thrust::copy(res_id_vec.begin(), res_id_vec.end(), std::ostream_iterator<int>(std::cout, ","));
}

int comdb::select_all_id(int *result)
{
	thrust::device_vector<int> id_vec(size);
	thrust::stable_sort(col_id_vec.begin(), col_id_vec.end());
	
	// if not two neighbor id equal, put in
	get_all_id gad(thrust::raw_pointer_cast(id_vec.data()), thrust::raw_pointer_cast(col_id_vec.data()));
	thrust::for_each(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(size), gad);	

	//thrust::copy_n(id_vec.begin(), id_vec.end(), std::ostream_iterator<int>(std::cout, ","));

	auto end_ptr = thrust::remove_if(id_vec.begin(), id_vec.end(), thrust::placeholders::_1 == -1);
	
	int length = end_ptr - id_vec.begin();
	thrust::copy_n(id_vec.begin(), length, result);
	
	//thrust::copy_n(result, length, std::ostream_iterator<int>(std::cout, ","));

	return length;
}

int comdb::select_by_time(const char *start, const char *end)
{
	// parse time into long value
	struct tm tm_start, tm_end;
	time_t t_start, t_end;
	
	if (strptime(start, "%Y-%m-%d %H:%M:%S", &tm_start) == NULL) 
	{
		perror("parse date error!");
		return EXIT_FAILURE;
	}
	
	if (strptime(end, "%Y-%m-%d %H:%M:%S", &tm_end) == NULL)
	{
		perror("parse date error!");
		return EXIT_FAILURE;
	}
	
	// we are at GMT-8, so add 8 hours
	t_start = mktime(&tm_start) + 8 * 3600;
	t_end = mktime(&tm_end) + 8 * 3600;

	time_between tbn(t_start, t_end);
	
	size_t _size = col_time_vec.size();
	res_lat_vec.resize(_size);

	thrust::for_each(
		thrust::make_zip_iterator(thrust::make_tuple(col_time_vec.begin(), col_lat_vec.begin(), res_lat_vec.begin())),
	    	thrust::make_zip_iterator(thrust::make_tuple(col_time_vec.end(), col_lat_vec.end(), res_lat_vec.end())),
		tbn);
	return col_time_vec.end() - col_time_vec.begin();
}


int comdb::select_by_space(double top_left_lon, double top_left_lat, double bottom_right_lon, double bottom_right_lat)	
{
 
	space_between sbn(top_left_lon, top_left_lat, bottom_right_lon, bottom_right_lat);
	
	// resize result vector equal to the size of data columns
	size_t _size = col_lon_vec.size();
	res_id_vec.resize(_size);	
	thrust::fill(res_id_vec.begin(), res_id_vec.end(), -1);

	thrust::for_each(
		thrust::make_zip_iterator(thrust::make_tuple(col_lon_vec.begin(), col_lat_vec.begin(), 
			col_id_vec.begin(), res_id_vec.begin())),
		thrust::make_zip_iterator(thrust::make_tuple(col_lon_vec.end(), col_lat_vec.end(), 
			col_id_vec.end(), res_id_vec.end())),
		sbn
	); 	

	//	std::cout << "----------- line 257" << std::endl;
	

  	//thrust::copy(res_id_vec.begin(), res_id_vec.end(), std::ostream_iterator<int>(std::cout, "\n"));

	size = res_id_vec.end() - res_id_vec.begin();
	return size;
}

int comdb::select_by_space_time(double top_left_lon, double top_left_lat, double bottom_right_lon, 
				double bottom_right_lat, long start, long end)
{
	space_time_between stb(top_left_lon, top_left_lat, bottom_right_lon, bottom_right_lat, start, end);
	
	size_t _size = col_lon_vec.size();
	res_id_vec.resize(_size);

	thrust::for_each(
		thrust::make_zip_iterator(thrust::make_tuple(col_lon_vec.begin(), col_lat_vec.begin(), 
			col_time_vec.begin(), col_id_vec.begin(), res_id_vec.begin())),
		thrust::make_zip_iterator(thrust::make_tuple(col_lon_vec.end(), col_lat_vec.end(),
			col_time_vec.end(), col_id_vec.end(), res_id_vec.end())),
		stb
	);

	return res_id_vec.end() - res_id_vec.begin();
}
