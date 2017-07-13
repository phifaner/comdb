// hash a trajectory id and a date into a long value
unsigned long hash_file(char * tid, char * date);

// read index file and store on CPU memory
void read_index(const char * filename);

// get file path given a trajectory id and a date
const char * get_path(char * tid, char * date);
