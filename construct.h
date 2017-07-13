#ifndef _CONSTRUCT_H
#define _CONSTRUCT_H
#include <vector>

#include "s2/base/basictypes.h"

using namespace std;

class Trajectory
{
    // trajectory id
    int tid;

    // points 
    double *lon, *lat;
    unsigned long *ts;
    int m_num;

  public: 
    Trajectory( int n )
    {
        lon = new double[n];
        lat = new double[n];
        ts  = new unsigned long[n];
        m_num = n;
    }

    inline void set_id( int id ) { tid = id; }

    inline void add_lon( double l, int idx ) { lon[idx] = l; }

    inline void add_lat( double l, int idx ) { lat[idx] = l; }

    inline void add_ts( long t, int idx ) { ts[idx] = t; }

    inline int get_tid() { return tid; }

    inline double *get_lon() { return lon; }
    
    inline double *get_lat() { return lat; }

    inline unsigned long *get_ts() { return ts; }

    inline int num() { return m_num; }
};

// trajectory points after construct
typedef struct T_Item
{
    uint64          cell_id;    // cell id in 64 bits
    unsigned int    tid;        // trajectory id
    unsigned long   ts;         // timestamp of start
} Item;

// std::ostream& operator<<(std::ostream& s, const Item& p)
// {
//     return s << p.cell_id ;
// }

class Construct
{
  public:
    Construct() 
    { 
        num_file = 0; 
        for (int i = 0; i < 120000; i++)  file_array[i] = (char *) malloc(100);
    }

    // check if a line intersect a rectangle
    bool intersect( double *, double *, double *, double *);

    // on level l, assign a trajectory's line segment into cells
    vector<Item> assign_line( unsigned int tid, double *lon, double *lat, unsigned long *ts, int l );

    // assign a trajectory into cells
    vector<Item> assign( Trajectory *traj, int l );

    // load trajectory from files
    Trajectory load_file( const char * path );

    // return s2 level
    int find_files( const char *path);

    char ** get_file_array() { return file_array; }

    int get_file_num() { return num_file; }

    // output new points in files
    void save_cells(vector<Item> &cells, char *path);

  private:

    Trajectory parse_file_msra( char *, size_t );
    
    Trajectory parse_file_wz( char *, size_t );
    
    // count number of lines
    int count_line( char *, size_t );
 
    // parser timestamp
    long atots( char * );

    uint64 cell_id;
    
    // for file names
    char *file_array[120000]; 
    int num_file;

   //     
};


#endif
