#include <thrust/device_vector.h>
#include <thrust/count.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>

struct is_equal_count
{
  int *tid_data;
  int *count;

  is_equal_count(int *tid, int *c) : tid_data(tid), count(c) {}

  __host__ __device__
  void operator() (const int & i)
  {
    if (tid_data[i] != tid_data[i+1])
      count[i] = 1;
  }

};

// struct position
// {
//   int *tid_data;
//   int *pos_data;
//   int len;

//   __host__ __device__
//   position() {}

//   template <typename T>
//   __host__ __device__
//   void operator() (const T& t)
//   {

//   }
// };

// output a stream of string
struct make_stream
{
  unsigned int *index;
  int *tid_data;
  double *lon_data, *lat_data;
  char **res_data;

  __host__ __device__
  make_stream(unsigned int *len, int *tid, double *lon, double *lat, char** res) 
    : index(len), tid_data(tid), lon_data(lon), lat_data(lat), res_data(res) {}

  __host__ __device__
  int itoa(int i, char *a)
  {
    int len = 0;
    int L = 0;
    int I = i;
    while (I > 0) 
    {
      L++;
      I /= 10;
    }

    len = L;
    a[L--] = '\0';

    for (; i >= 10; --L)
    {
      a[L] = '0' + i % 10;

      i /= 10;
    }

    a[L] = '0' + i;

                // printf("%d, %s\n", len, a);
    return len;
  }

  __host__ __device__
  int floor(double d)
  {
    int I = (int)d;

    if (d - I > 0.5) return I+1;
    else return I;
  }

  __host__ __device__
  int dtoa(double d, char *a)
  {
    int L = 0, S;
    double I = d;
    while (I >= 1) 
    {
      L++;
      I /= 10;
    }

    S = L++;

    int U;
    I = d - (int)(d);
    I *= 10;
    U = (int) (I);

    printf("%lf, %lf\n", d, d - (int)(d));

    // error
    while (I > 0 && L < 8)
    {
      a[L] = '0' + U;

      L++;
      I = I - (int)(I);
      I *= 10;
      U = (int)(I);
    }


    a[L] = '\0';
    a[S--] = '.';

    I = d;
    // tackle LEFT part of '.'
    for (; I >= 10; --S)
    {
      a[S] = '0' + ((int)I)% 10;
      I /= 10;
    }

    a[0] = '0' + (int)I;

    // printf("-------%s\n", a);

    return L;
  }

  __host__ __device__
  void cuda_concat(char *a, int len_a, char *b, int len_b)
  {

    for (int idx = 0; idx < len_b; ++idx)
    {
        a[len_a+idx] = b[idx];
    }

  }


  template <typename T>
  __device__
  void operator() (const T& i)
  {

    if (tid_data[i] != tid_data[i+1] || i == 0)
    {
      atomicAdd(index+1, 1);
            char *sstring = res_data[index[1]];
             printf("-----before: %d\n", index[1]);
      __syncthreads();
printf("-----after: %d\n", index[1]);
      // __syncthreads();
      // char *sstring = new char[100];
      int k = (i==0 ? i : i+1);
      // char *s = new char[20];
      char *s = "{\"trip\": ";
      cuda_concat(sstring, 0, s, 9);

      int len = 9;
      // char *ss = new char[20];
      char ss[20];
      int _l = itoa(tid_data[k], ss);
      cuda_concat(sstring, len, ss, _l);
      len += _l;
      // delete [] ss;

      // sstring[len] = '\0';
    // printf("%d, %s\n", len, sstring);

      char str[20] = ", \"points\": [";
      cuda_concat(sstring, len, str, 13);
      len += 13;

      for (; tid_data[k] == tid_data[k+1]; ++k)
      {
        cuda_concat(sstring, len, "[", 1);
        len += 1;

        // char *sss = new char[20];
        char sss[20];
        int l = dtoa(lon_data[k], sss);
        cuda_concat(sstring, len, sss, l);
        len += l;
        // delete [] sss;


        cuda_concat(sstring, len, ", ", 2);
        len += 2;

        // char *ssss = new char[20];
        char ssss[20];
        l = dtoa(lat_data[k], ssss);
        cuda_concat(sstring, len, ssss, l);
        len += l;
        // delete [] ssss;

        cuda_concat(sstring, len, "], ", 3);
        len += 3;

        // printf("============%d, %d\n", tid_data[k], tid_data[k+1]);
        // printf("============%d, %s\n", i, sstring);
      }

      // to handle the last item
      cuda_concat(sstring, len, "[", 1);
      len += 1;

      int l = dtoa(lon_data[k], s);

      cuda_concat(sstring, len, s, l);
      len += l;

      cuda_concat(sstring, len, ", ", 2);
      len += 2;

      l = dtoa(lat_data[k], s);
      cuda_concat(sstring, len, s, l);
      len += l;

      cuda_concat(sstring, len, "], ", 3);
      len += 3;

      // finish the json stream
      cuda_concat(sstring, len, "[0, 0]]", 8);
      len += 7;

      sstring[len] = '\0';
       // printf("---------------- %s\n", sstring);              
      // delete [] s;
      //  atomicAdd(index+1, 1);
      // __syncthreads();
    }
  }
};

struct check_stream
{
  size_t MAX;
  size_t *position_data;
  int *tid_data;
  double *lon_data, *lat_data;
  char **res_data;

  __host__ __device__
  check_stream(size_t *position, size_t max, int *tid, double *lon, double *lat, char** res) 
    : position_data(position), MAX(max), tid_data(tid), lon_data(lon), lat_data(lat), res_data(res) {}

  __host__ __device__
  int itoa(int i, char *a)
  {
    int len = 0;
    int L = 0;
    int I = i;
    while (I > 0) 
    {
      L++;
      I /= 10;
    }

    len = L;
    a[L--] = '\0';

    for (; i >= 10; --L)
    {
      a[L] = '0' + i % 10;

      i /= 10;
    }

    a[L] = '0' + i;

                // printf("%d, %s\n", len, a);
    return len;
  }

  __host__ __device__
  int dtoa(double d, char *a)
  {
    int L = 0, S;
    double I = d;
    while (I >= 1) 
    {
      L++;
      I /= 10;
    }

    S = L++;

    int U;
    I = d - (int)(d);
    I *= 10;
    U = (int) (I);

    // printf("%lf, %lf\n", d, d - (int)(d));

    // error
    while (I > 0 && L < 8)
    {
      a[L] = '0' + U;

      L++;
      I = I - (int)(I);
      I *= 10;
      U = (int)(I);
    }


    a[L] = '\0';
    a[S--] = '.';

    I = d;
    // tackle LEFT part of '.'
    for (; I >= 10; --S)
    {
      a[S] = '0' + ((int)I)% 10;
      I /= 10;
    }

    a[0] = '0' + (int)I;

    // printf("-------%s\n", a);

    return L;
  }

  __host__ __device__
  void cuda_concat(char *a, int len_a, char *b, int len_b)
  {

    for (int idx = 0; idx < len_b; ++idx)
    {
        a[len_a+idx] = b[idx];
    }

  }

  __host__ __device__
  void operator() (const int & i)
  {
    if (position_data[i] < MAX && position_data[i] > 0 || i == 0)
    {
      size_t k = i == 0 ? 0 : position_data[i]+1;
      char * sstring = res_data[i];

      // start to output a new trajectory
      char *s = "{\"trip\": ";
      cuda_concat(sstring, 0, s, 9);

      int len = 9;
      // char *ss = new char[20];
      char ss[20];
      int _l = itoa(tid_data[k+1], ss);
      cuda_concat(sstring, len, ss, _l);
      len += _l;
      // delete [] ss;

      // sstring[len] = '\0';
    // printf("%d, %s\n", len, sstring);

      char str[20] = ", \"points\": [";
      cuda_concat(sstring, len, str, 13);
      len += 13;

      for (; k < position_data[i+1]; ++k)
      {
        cuda_concat(sstring, len, "[", 1);
        len += 1;

        // char *sss = new char[20];
        char sss[20];
        int l = dtoa(lon_data[k], sss);
        cuda_concat(sstring, len, sss, l);
        len += l;
        // delete [] sss;


        cuda_concat(sstring, len, ", ", 2);
        len += 2;

        // char *ssss = new char[20];
        char ssss[20];
        l = dtoa(lat_data[k], ssss);
        cuda_concat(sstring, len, ssss, l);
        len += l;
        // delete [] ssss;

        cuda_concat(sstring, len, "], ", 3);
        len += 3;

        // printf("============%d, %d\n", tid_data[k], tid_data[k+1]);
        // printf("============%d, %s\n", i, sstring);
      }

      // to handle the last item
      cuda_concat(sstring, len, "[", 1);
      len += 1;

      char s4[20];
      int l = dtoa(lon_data[k], s4);

printf("============%lu, %lf\n", k, lon_data[k]);

      cuda_concat(sstring, len, s4, l);
      len += l;

      cuda_concat(sstring, len, ", ", 2);
      len += 2;

      char s5[20];
      l = dtoa(lat_data[k], s5);
      cuda_concat(sstring, len, s5, l);
      len += l;

      cuda_concat(sstring, len, "], ", 3);
      len += 3;

      // finish the json stream
      cuda_concat(sstring, len, "[0, 0]]", 8);
      len += 7;

      sstring[len] = '\0';
    }
  }
  
};

int main()
{

  std::vector<int> h_tid_vec(6);
  h_tid_vec[0] = 98;
  h_tid_vec[1] = 98;
  h_tid_vec[2] = 98;
  h_tid_vec[3] = 86;
  h_tid_vec[4] = 86;
  h_tid_vec[5] = 86;

  std::vector<double> h_lon_vec(6);
  h_lon_vec[0] = 40.1162;
  h_lon_vec[1] = 39.9399;
  h_lon_vec[2] = 39.9401;
  h_lon_vec[3] = 121.123;
  h_lon_vec[4] = 812.123;
  h_lon_vec[5] = 82.23;

  thrust::device_vector<int> tid_vec = h_tid_vec;
  thrust::device_vector<double> lon_vec = h_lon_vec;
  thrust::device_vector<double> lat_vec = h_lon_vec;
  thrust::device_vector<char *> res_vec(2);

  thrust::device_vector<int>      holder_vec(6, 0);
  thrust::device_vector<size_t>   position_vec(6, 0);
  holder_vec[0] = 1;
  holder_vec[5] = 1;

  is_equal_count iec(thrust::raw_pointer_cast(tid_vec.data()), thrust::raw_pointer_cast(holder_vec.data()) );

  thrust::for_each(thrust::make_counting_iterator(0), thrust::make_counting_iterator(6), iec);


  thrust::copy_if(thrust::make_counting_iterator(0), thrust::make_counting_iterator(6), holder_vec.begin(), position_vec.begin(),  thrust::placeholders::_1 == 1);

  thrust::copy(position_vec.begin(), position_vec.end(), std::ostream_iterator<size_t>(std::cout, ","));

  for (int i = 0; i < 2; i++)
  {
    // thrust::device_vector<char> dev_str(3, 0);
    // res_vec[i] = thrust::raw_pointer_cast(dev_str.data());
    // char *d_str;
    // cudaMalloc((void**)&d_str, 3*sizeof(char)); 

    char *device_array_a = 0 ;
    cudaMalloc((void**)&device_array_a, 200);
    res_vec[i] = device_array_a;
  }

  check_stream cs
  (
    thrust::raw_pointer_cast(position_vec.data()),
    5,
    thrust::raw_pointer_cast(tid_vec.data()),
    thrust::raw_pointer_cast(lon_vec.data()),
    thrust::raw_pointer_cast(lat_vec.data()),
    thrust::raw_pointer_cast(res_vec.data())
  );

  thrust::for_each(thrust::make_counting_iterator(0), thrust::make_counting_iterator(6), cs);
  // thrust::device_vector<unsigned int > index(2);
  // thrust::sequence(index.begin(), index.end());
  // make_stream ms
  // (
  //   thrust::raw_pointer_cast(index.data()),
  //   thrust::raw_pointer_cast(tid_vec.data()),
  //   thrust::raw_pointer_cast(lon_vec.data()),
  //   thrust::raw_pointer_cast(lat_vec.data()),
  //   thrust::raw_pointer_cast(res_vec.data())
  // );

  // thrust::for_each(thrust::make_counting_iterator(0), thrust::make_counting_iterator(5), ms);

  for (int i = 0; i < 2; i++)
  {
        // thrust::host_vector<char*> h_res( &res_vec[0], &res_vec[0] + 2 );
        char *h_array_a = (char *)calloc(200, 1);
        cudaMemcpy(h_array_a, res_vec[i], 200, cudaMemcpyDeviceToHost);
        printf("-----%s\n", h_array_a);
  }
}