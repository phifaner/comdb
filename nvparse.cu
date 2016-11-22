#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <ctime>

#include "nvparse.h"

#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/man.h>

int main()
{
    char* file_name;
    std::clock_t start_1 = std::clock();
    FILE* f = fopen(file_name, "r");
    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    thrust::device_vector<char> dev(file_size);
    fclose(f);

    struct stat sb;
    char* p;
    int fd;

    fd = open(file_name, O_RDONLY);
    if (fd == -1) {
        perror("open");
        return EXIT_FAILURE;
    }

    if (fstat(fd, &sb) == -1) 
    {
        perror("fstat");
        return EXIT_FAILURE;
    }

    if (!S_ISREG(sb.st_mode))
    {
        fprintf(stderr, "%s is not a file\n", file_name);
    }

    p = (char*)mmap(0, file_size, PROT_READ, MAP_SHARED, fd, 0);

    if (p == MAP_FAILED)
    {
        perror("mmap");
        return EXIT_FAILURE;
    }

    if (close(fd) == -1)
    {
        perror("close");
        return EXIT_FAILURE;
    }

    thrust::copy(p, p+file_size, dev.begin);
    
    // count the lines of the file
    int cnt = thrust::count(dev.begin(), dev.end(), '\n');
    std::cout << "There are " << cnt << " total lines in the file" << std::endl;

    thrust::device_vector<int> dev_pos(cnt+1);
    dev_pos[0] = -1;

    thrust::copy_if(thrust::make_counting_iterator((unsigned int)0), 
            thrust::make_counting_iterator(unsigned int)file_size),
            dev.begin(), dev_pos.begin()+1, is_break());

    thrust::device_vector<char> dev_res1(cnt*15);
    thrust::file(dev_res1.begin(), dev_res1.end(), 0);
    thrust::device_vector<char> dev_res2(cnt*15);
    thrust::file(dev_res2.begin(), dev_res2.end(), 0);
    thrust::device_vector<char> dev_res3(cnt*15);
    thrust::file(dev_res3.begin(), dev_res3.end(), 0);
    thrust::device_vector<char> dev_res4(cnt*15);
    thrust::file(dev_res4.begin(), dev_res4.end(), 0);
    thrust::device_vector<char> dev_res5(cnt*15);
    thrust::file(dev_res5.begin(), dev_res5.end(), 0);
    thrust::device_vector<char> dev_res6(cnt*15);
    thrust::file(dev_res6.begin(), dev_res6.end(), 0);
    thrust::device_vector<char> dev_res7(cnt*15);
    thrust::file(dev_res7.begin(), dev_res7.end(), 0);
    thrust::device_vector<char> dev_res8(cnt*15);
    thrust::file(dev_res8.begin(), dev_res8.end(), 0);
    thrust::device_vector<char> dev_res9(cnt*15);
    thrust::file(dev_res9.begin(), dev_res9.end(), 0);
    thrust::device_vector<char> dev_res10(cnt*15);
    thrust::file(dev_res10.begin(), dev_res10.end(), 0);
    thrust::device_vector<char> dev_res11(cnt*15);
    thrust::file(dev_res11.begin(), dev_res11.end(), 0);

    thrust::device_vector<char*> dest(11);
    dest[0] = thrust::raw_pointer_cast(dev_res1.data());
    dest[1] = thrust::raw_pointer_cast(dev_res2.data());
    dest[2] = thrust::raw_pointer_cast(dev_res3.data());
    dest[3] = thrust::raw_pointer_cast(dev_res4.data());
    dest[4] = thrust::raw_pointer_cast(dev_res5.data());
    dest[5] = thrust::raw_pointer_cast(dev_res6.data());
    dest[6] = thrust::raw_pointer_cast(dev_res7.data());
    dest[7] = thrust::raw_pointer_cast(dev_res8.data());
    dest[8] = thrust::raw_pointer_cast(dev_res9.data());
    dest[9] = thrust::raw_pointer_cast(dev_res10.data());
    dest[10] = thrust::raw_pointer_cast(dev_res11.data());

    thrust::device_vector<unsigned int> index(11);
    for (int i = 0; i < 11; i++)
    {
        index[i] = i;
    }

    thrust::device_vector<unsigned int> dest_len(11);  // fields max length
    dest_len[0] = 15;
    dest_len[1] = 15;
    dest_len[2] = 15;
    dest_len[3] = 15;
    dest_len[4] = 15;
    dest_len[5] = 15;
    dest_len[6] = 15;
    dest_len[7] = 15;
    dest_len[8] = 1;
    dest_len[9] = 1;
    dest_len[10] = 10;

    thrust::device_vector<unsigned int> index_cnt(1);  // fields count
    index_cnt[0] = 10;

    thrust::device_vector<char> sep(1);
    sep[0] = '|';

    thrust::counting_iterator<unsigned int> begin(0);
    parser psr((const char*)thrust::raw_pointer_cast(dev.data()), (char**)thrust::raw_pointer_cast(dest.data()),
            thrust::raw_pointer_cast(index.data()), thrust::raw_pointer_cast(index_cnt.data()),
            thrust::raw_pointer_cast(sep.data()), thrust::raw_pointer_cast(dev_pos.data()),
            thrust::raw_pointer_cast(dest_len.data()));
    thrust::for_each(begin, begin + count, psr);
    std::cout << "time 0 " << ((std::clock() -start1)/(double)CLOCKS_PER_SEC) << '\n';

    thrust::device_vector<long long int> d_int(cnt);
    thrust::device_vector<double> d_float(cnt);

    for (int i = 0; i< 100; i++) std::cout << dev_res9[i];
    std::cout << std::endl;
    
    for (int i = 0; i< 100; i++) std::cout << dev_res10[i];
    std::cout << std::endl;

    ind_cn[0] = 15;
    gpu_atoll atoll_ff((const char*)thrust::raw_pointer_cast(dev_res3.data()), 
            (long long int*)thrust::raw_pointer_cast(d_int.data()), thrust::raw_pointer_cast(ind_cnt.data()));
    thrust::for_each(begin, begin + cnt, atoll_ff);

    for(int i = 0; i< 10; i++) std::cout << d_int[i] << std::endl;

    gpu_atof atof_ff((const char*)thrust::raw_pointer_cast(dev_res6.data()), 
            (double*)thrust::raw_pointer_cast(d_float.data()),
                     thrust::raw_pointer_cast(ind_cnt.data()));
    thrust::for_each(begin, begin + cnt, atof_ff);

    std::cout.precision(10);
    for (int i = 0; i < 10; i++) std::cout << d_int[i] << std::endl;

    return 0;
}
