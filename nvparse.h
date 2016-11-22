struct is_break
{
    __host__ __device__ boolean operator () (const char x)
    {
        return x == 10;
    }
};

struct parser
{
    const char *source;
    char **dest;
    const unsigned int *index;
    const unsigned int *count;
    const char *separator;
    const int *src_index;
    const unsigned int dest_len;

    parser(const char* _source, char** _dest, const unsigned int* _index,
            const unsigned int* _count, const char* _separator,
            const int* _src_index, const unsigned int* _dest_len):
        source(_source), dest(_dest), index(_index), count(_count),
        separator(_separator), src_index(_src_index), dest_len(_dest_len) {}

    template <typename IndexType> __host__ __device__
    void operator() (const IndexType &i)
    {
        unsigned int current_count = 0, dest_current = 0, j = 0, t, pos;
        pos = src_index[i]+1;

        while (dest_current < *count)
        {
            if (index[dest_current] == current_count)
            {
                t = 0;
                while (source[pos+j] != separator)
                {

                }
            }
        }
    }
}

