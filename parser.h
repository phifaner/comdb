#include "comdb.h"

struct DataLoader
{
	/* load multiple files by trajectory ids,
	 * len: length of list
	 */
	int load_data_by_ids(const unsigned long * id_list, int len, comdb *db);

	/* load only one file */
	int load_data_wz(const char * filename, comdb * db);

	/* load data of beijing taxi */
	int load_data_bj(const char* filename, comdb &db);

	/* get all files of a directory */
	std::vector<char*> find_files( const char* folder );
};

struct is_break
{
    __host__ __device__
    bool operator() (const char x)
    {
        return x == 10;
    }
};

struct parser
{
    const char *source;
    char **dest;
    const unsigned int *index;
    const unsigned int *cnt;            	// amount of fields
    const char *separator;
    const int *src_line_index;               	// line cursor of source data
    const unsigned int *dest_len;		// each destion column length

    parser(const char* _source, char** _dest, const unsigned int* _index,
            const unsigned int* _count, const char* _separator,
            const int* _src_line_index, const unsigned int* _dest_len):
        source(_source), dest(_dest), index(_index), cnt(_count),
        separator(_separator), src_line_index(_src_line_index), dest_len(_dest_len) {}

    template <typename T>
   __host__  __device__
    void operator() (const T& i)
    {
        unsigned int curr_cnt = 0, dest_curr = 0, j = 0, t, pos;
        pos = src_line_index[i]+1;

        // scan each field
        while (dest_curr < *cnt)
        {
            // only deal with columns in the index array
            if (index[dest_curr] == curr_cnt)
            {
		t = 0;

                // if not meeting separator, copy char from source to dest
                while (source[pos+j] != *separator && t < dest_len[dest_curr])
                {
                    if (source[pos+j] != 0)
                    {
                        dest[dest_curr][dest_len[dest_curr]*i+t] = source[pos+j];
			t++;
                    }
                    j++;
                }
                j++;            // jump separator
                dest_curr++;    // deal with the next column value
            }
            else
            {
                while (source[pos+j] != *separator) j++;
            }

            curr_cnt++;
        }
//	__syncthreads();
    }
};

struct gpu_atof
{
    const char *source;
    double *dest;
    const unsigned int *len;

    gpu_atof(const char *_source, double *_dest, const unsigned int *_len):
	source(_source), dest(_dest), len(_len) {}

    template<typename T>
    __host__ __device__
    void operator() (const T &i)
    {
	const char *p;
	int frac;
	double sign, value, scale;
	
	p = source + len[0]*i;
	const char *end = p + len[0];

	while (*p == ' ')
	{
	    p += 1;
	}

	sign = 1.0;
	if (*p == '-')
	{
	    sign = -1.0;
	    p += 1;
	}
	else if (*p == '+')
	{
	    p += 1;
	}
	
	for (value=0.0; *p>='0'&&*p<='9'; p++)
	{
	    value = value * 10.0 + (*p - '0');
	}
	
	// deal with fraction part	
	if (*p == '.')
	{
	    double pow10 = 10.0;
	    p += 1;
	    while (p<end && *p>='0' && *p<='9')
	    {
 		value += (*p-'0')/pow10;
		pow10 *= 10.0;
		p += 1;

	    }
	}
	
	frac = 0;
	scale = 1.0;

	dest[i] = sign * (frac ? (value/scale) : (value*scale));
	
	//if (dest[i] > 1000)
	//	printf("atof------------%f: %s \n", dest[i], source);
    }
};

struct gpu_atoi
{
	const char *source;
	int *dest;
	const unsigned int *len;

	gpu_atoi(const char *_source, int *_dest, const unsigned int *_len):
		source(_source), dest(_dest), len(_len) {}

	template<typename T>
	__host__ __device__
	void operator() (const T &i)
	{
		const char *p;
		int frac, sign, value, scale;
		

		p = source + len[0]*i;
		const char *end = p + len[0];

        while (*p == ' ' || *p == '\n' || *p == '\t')
		{
			p += 1;
		}

		sign = 1;
		if (*p == '-')
		{
			sign = -1;
			p += 1;
		}
		else if (*p == '+')
		{
			p += 1;
		}
		
		for (value = 0; *p >= '0' && *p <= '9' && p < end; p += 1)
		{
			value = value * 10 + (*p - '0');
		}
		

		frac = 0;
		scale = 1;
		
		dest[i] = sign * (frac ? (value/scale) : (value*scale));

        //if (dest[i] != 101)
//printf("+++++++++++ v: %d, i: %d\n", i, dest[i]);
	}
};

struct gpu_atoul
{
	const char *source;
	unsigned long *dest;
	const unsigned int *len;

	gpu_atoul(const char *_source, unsigned long *_dest, const unsigned int *_len):
		source(_source), dest(_dest), len(_len) {}

	template<typename T>
	__host__ __device__
	void operator() (const T &i)
	{
		const char *p;
		unsigned long long value;
		

		p = source + len[0]*i;
		const char *end = p + len[0];

        while (*p == ' ' || *p == '\n' || *p == '\t')
		{
			p += 1;
		}

		/*sign = 1;
		if (*p == '-')
		{
			sign = -1;
			p += 1;
		}
		else if (*p == '+')
		{
			p += 1;
		}*/
		
		for (value = 0; *p >= '0' && *p <= '9' && p < end; p += 1)
		{
			value = value * 10 + (*p - '0');
		}
		
		dest[i] = value;

	}
};


struct gpu_atoull
{
	const char *source;
	unsigned long long *dest;
	const unsigned int *len;

	gpu_atoull(const char *_source, unsigned long long *_dest, const unsigned int *_len):
		source(_source), dest(_dest), len(_len) {}

	template<typename T>
	__host__ __device__
	void operator() (const T &i)
	{
		const char *p;
		unsigned long long value;
		

		p = source + len[0]*i;
		const char *end = p + len[0];

        while (*p == ' ' || *p == '\n' || *p == '\t')
		{
			p += 1;
		}

		/*sign = 1;
		if (*p == '-')
		{
			sign = -1;
			p += 1;
		}
		else if (*p == '+')
		{
			p += 1;
		}*/
		
		for (value = 0; *p >= '0' && *p <= '9' && p < end; p += 1)
		{
			value = value * 10 + (*p - '0');
		}
		
		dest[i] = value;

        //if (dest[i] != 101)
//printf("+++++++++++ v: %d, i: %d\n", i, dest[i]);
	}
};



// parse date format: YYYY-MM-DD hh:mm:ss
// return a second value
struct gpu_date
{
	const char *source;
	long int *dest;
	const unsigned int *len;
	
	gpu_date(const char *_source, long int *_dest, const unsigned int *_len):
		source(_source), dest(_dest), len(_len) {}
	
	template <typename T>
	__host__ __device__
	void operator() (const T &i)
	{
		const char *s;
		int year, month, day, hour, min, second;
		long int acc;
		int z = 0, c;

		s = source + len[0]*i;
		//c = (unsigned char) *s++;
		

		for (year = 0; z < 4; )
		{
			c = (unsigned char) *s++;
		
			if (c != '-') 
			{
				c -= '0';
				year *= 10;
				year += c;	
			}
			z++;
		}
		
		// jump '-'
		c = *s++;
		z = 0; 

		for (month = 0; z < 2; )
		{
			c = (unsigned char) *s++;
			c -= '0';
			month *= 10;
			month += c;
			z++;
		}

		// jump '-'
		c = *s++;
		z = 0;

		for (day=0; z < 2; )
		{
			c = (unsigned char) *s++;
			c -= '0';
			day *= 10;
			day += c;
			z++;
		}

		// jump ' '
		c = *s++;
		z = 0;

		for (hour = 0; z < 2; )
		{
			c = (unsigned char) *s++;
			c -= '0';
			hour *= 10;
			hour += c;
			z++;
		}
		
		// jump ':'
		c = *s++;
		z = 0;

		for (min = 0; z < 2; )
		{
			c = (unsigned char) *s++;
			c -= '0';
			min *= 10;
			min += c;
			z++;
		}

		// jump ':'
		c = *s++;
		z = 0;

		for (second = 0; z < 2; )
		{
			c = (unsigned char) *s++;
			c -= '0';
			second *= 10;
			second += c;
			z++;
		}
		
		// evaluating leap years from 1970 to the year
		int cnt = 0;
		int from_year = 1970;
		int to_year = year;
		
		while (from_year < to_year)
		{
			if ((from_year%4==0 && from_year%100!=0) || (from_year%400==0)) cnt++;
			
			from_year++;
		}
		
		// evaluating time in second, be careful of Time Zone - 8
		acc = (year - 1970) * 365 * 24 * 3600 + (cnt+day-1) * 24 * 3600 + (hour-8) * 3600 + min * 60 + second;
		int feb = ((year%4==0&&year%100!=0) || (year%400==0)) ? 29 * 24 * 3600 : 28 * 24 * 3600;

		switch (month)
		{
			case 1: break;
			case 2: acc += 31 * 24 * 3600;		// value of January + 31 days
				break;
			case 3: acc += 31 * 24 * 3600 + feb;
				break;
			case 4: acc += 2 * 31 * 24 * 3600 + feb;
				break;
			case 5: acc += (2 * 31 + 30)  * 24 * 3600 + feb;
				break;
			case 6: acc += (3 * 31 + 30) * 24 * 3600 + feb;
				break;
			case 7: acc += (3 * 31 + 2 * 30) * 24 * 3600 + feb;
				break;
			case 8: acc += (4 * 31 + 2 * 30) * 24 * 3600 + feb;
				break;
			case 9: acc += (5 * 31 + 2 * 30) * 24 * 3600 + feb;
				break;
			case 10: acc += (5 * 31 + 3 * 30) * 24 * 3600 + feb;
				break;
			case 11: acc += (6 * 31 + 3 * 30) * 24 * 3600 + feb;
				break;
			case 12: acc += (6 * 31 + 4 * 30) * 24 * 3600 + feb;
				break;
			default: break;
		} 
		
		dest[i] = acc;
	}	
};
