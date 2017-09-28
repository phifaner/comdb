#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <stdint.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <unistd.h>
#include <dirent.h>

#include "s2/s2latlng.h"
#include "s2/s2cellid.h"
#include "s2/s2cell.h"
#include "s2/s2latlngrect.h"

#include "construct.h"

// decide if a line segment intersect a rectangle
// f(x) = a(x-xm)+b(y-ym)
// if all corners are >0 or <0, no possible
bool Construct::intersect( double *x, double *y, double *ll, double *ur )
{
    // project the endpoint onto the x and y axis
    // check if the segment's shadow intersect the rectangle's shadow
    if ( min(x[0], x[1]) > max(ll[0], ur[0]) ) return false;    // right
    if ( max(x[0], x[1]) < min(ll[0], ur[0]) ) return false;    // left
    if ( min(y[0], y[1]) > max(ll[1], ur[1]) ) return false;    // top
    if ( max(y[0], y[1]) < min(ll[1], ur[1]) ) return false;    // below

    // check if all four corners are on the same side of the segment
    double xm = (x[0]+x[1])/2;
    double ym = (y[0]+y[1])/2;
    double a = y[1]-y[0];
    double b = x[0]-x[1];

    double f_ll = a * (ll[0]-xm) + b * (ll[1]-ym);       // lower left vertex
    double f_ul = a * (ll[0]-xm) + b * (ur[1]-ym);       // upper left
    double f_ur = a * (ur[0]-xm) + b * (ur[1]-ym);       // upper right
    double f_lr = a * (ur[0]-xm) + b * (ll[1]-ym);       // lower right

    //printf("ll: %f, ul: %f, ur: %f, lr: %f\n", f_ll, f_ul, f_ur, f_lr);

    if ( f_ll > 0 && f_ul > 0 && f_ur > 0 && f_lr > 0) return false;
    if ( f_ll < 0 && f_ul < 0 && f_ur < 0 && f_lr < 0) return false;

    return true;
}

vector<Item> Construct::assign_line(unsigned int tid, double *lon, double *lat, unsigned long *ts, int l )
{
    // get s2 latitude and longitude of starting point and end point
    S2LatLng lat_lng_s = S2LatLng::FromDegrees( lat[0], lon[0] );
    S2LatLng lat_lng_e = S2LatLng::FromDegrees( lat[1], lon[1] );

    // get s2 cell id of level l
    S2CellId id_s = S2CellId::FromLatLng( lat_lng_s );
    S2CellId id_e = S2CellId::FromLatLng( lat_lng_e );
    S2CellId id_l_s = id_s.parent( l );
    S2CellId id_l_e = id_e.parent( l );

    vector<Item> item_vec; 
    
    // only add the starting cell into vector
    Item st;
    st.cell_id   = id_l_s.id();
    st.tid       = tid;
    st.ts        = ts[0];
    item_vec.push_back(st);
    
    //printf( "latitude:%f, longitude: %f, -- cell end: %llu, %llu\n", lat[0], lon[0], id_l_s.id(), id_l_e.id() );


    if ( id_l_s == id_l_e ) return item_vec;

    //printf( "level:%d, cell start: %f, %f, -- cell end: %llu, %llu\n", id_l_e.level(), lat_lng_s.coords()[0], lat_lng_s.coords()[1], id_l_s.id(), id_l_e.id() );

    // get number of cells between starting and end points
    Vector2_d st_s = id_l_s.GetCenterST();
    Vector2_d st_e = id_l_e.GetCenterST();
    double size_st = S2CellId::GetSizeST( l );
    //printf("======= %lf, %lf\n", st_e[1], st_s[1]);
    int num_cell_x = ceil( abs(st_e[0] - st_s[0]) / size_st );
    int num_cell_y = ceil( abs(st_e[0] - st_s[0]) / size_st );
    int num_cell = max(num_cell_x, num_cell_y);

    // consecutive points at the same cell or in neighbor cells
    if (num_cell == 0 || num_cell == 1) return item_vec;
    
    long size_time = ( ts[1] - ts[0] ) / num_cell;

    S2CellId c_idx = id_l_s;
    if ( id_l_s > id_l_e )  // make id_l_s before id_l_e
    {
        id_l_s = id_l_e;
        id_l_e = c_idx;
        c_idx = id_l_s;
    }
    
    // count the number of interpolation cells
    int idx = 0;
    // record last cell point in case of multiple copies of one point
    long last_ts = 0;
    uint64 last_cell_id = 0;

    // verify whether the line intersect cells between s and e
    while ( c_idx < id_l_e )
    {
        // get c_idx corner coordinates
        S2Cell c( c_idx );
        S2LatLngRect rect = c.GetRectBound();
        S2LatLng v0 = rect.GetVertex(0);
        S2LatLng v1 = rect.GetVertex(1); 
        S2LatLng v2 = rect.GetVertex(2);
        S2LatLng v3 = rect.GetVertex(3);
       
        // check if the current line intersect cell c_idx
        double ll[2] = {v0.coords()[0] * 1000, v0.coords()[1] * 1000};
        double ul[2] = {v1.coords()[0] * 1000, v1.coords()[1] * 1000};
        double ur[2] = {v2.coords()[0] * 1000, v2.coords()[1] * 1000};
        double lr[2] = {v3.coords()[0] * 1000, v3.coords()[1] * 1000};
        //printf( "lower left: %f %f, upper left: %f %f, upper right: %f %f, lower right: %f %f\n", ll[0], ll[1], ul[0], ul[1], ur[0], ur[1], lr[0], lr[1] );

        // check intersection
        double x[2] = {lat_lng_s.coords()[0] * 1000, lat_lng_e.coords()[0] * 1000};
        double y[2] = {lat_lng_s.coords()[1] * 1000, lat_lng_e.coords()[1] * 1000};
        bool flag = intersect( x, y, ll, ur );
       
        if ( flag )
        {
            // interpolate time
            Vector2_d cen = c_idx.GetCenterST();
            long curr_time = ts[0] + ( cen[0] - st_s[0] ) / size_st * size_time;
            
            //printf("current time: %ld, ts 1: %ld, ts 0: %ld, time size: %ld\n ", curr_time, ts[1], ts[0], size_time);
            //printf( "starting center: %lf %lf, end center: %lf %lf, cell size %lf, cell num: %d\n", st_s[0], st_s[1], st_e[0], st_e[1], size_st, num_cell );

            Item t;
            t.cell_id   = c_idx.id();
            t.tid       = tid;
            t.ts        = curr_time;
            
            idx ++;

            // add item into vector
            //if ( idx == 10 ) item_vec.resize( 20 );
            if ( idx == 10 ) 
            {
                printf( "Exceed the limits! \n" );
                item_vec.clear();
                return item_vec;
            }
                
            // if last cell point not equal to the interpolated point
            //if ( last_cell_id != t.cell_id ) 
                item_vec.push_back( t );

            //last_ts = curr_time;
            last_cell_id = t.cell_id;
        }

        c_idx = c_idx.next(); 
        
    }

    //std::copy(item_vec.begin(), item_vec.end(), std::ostream_iterator<Item>(std::cout, ","));

    return item_vec;
}

vector<Item> Construct::assign( Trajectory *traj, int l)
{
    int tid = traj->get_tid();
    double * lon_vec = traj->get_lon();
    double * lat_vec = traj->get_lat();
    unsigned long * ts_vec = traj->get_ts();

    vector<Item> traj_cell;
    int sum = 0;

    for ( int i = 0; i < traj->num()-1; i++) 
    {
        //printf( "lon: %f, lat: %f, lon_2: %f, lat_2: %f\n", lon_vec[i], lat_vec[i], lon_vec[i+1], lat_vec[i+1] );
        double lon[2] = {lon_vec[i], lon_vec[i+1]};
        double lat[2] = {lat_vec[i], lat_vec[i+1]};
        unsigned long ts[2] = {ts_vec[i], ts_vec[i+1]};
        vector<Item> line_cell = assign_line( tid, lon, lat, ts, l );
        
        //printf("num %d vector size: %lu\n", i, line_cell.size());


        // combine cells of line segment into trajectory cells
        if (line_cell.size() > 0)
        {
            sum += line_cell.size();

            //printf("size before insert: %lu\n", traj_cell.size());
            traj_cell.insert(traj_cell.end(), line_cell.begin(), line_cell.end());
            line_cell.clear();

            //printf("size after clear: %lu, container's size: %lu\n", line_cell.size(), traj_cell.size());
        }

    }

    //printf("total number of vector: %lu, sum: %d\n", traj_cell.size(), sum);
    return traj_cell;
}

Trajectory Construct::load_file( const char * path )
{
    struct stat s;
    int fd = open( path, O_RDONLY );   
    int status = fstat( fd, &s );
    if ( status == -1 )
        perror( "Error! Getting the file size! " );

    size_t size = s.st_size;
    if ( size == 0 )
        fprintf( stderr, "Error: File is empty\n" );

    char *mapped = (char*) mmap( 0, size, PROT_READ, MAP_SHARED, fd, 0 );
    if ( mapped == MAP_FAILED )
    {
        close( fd );
        perror( "Error! Mmaping file! " );
        exit( EXIT_FAILURE );
    }
    
    // parse file to get trajectory
    Trajectory traj = parse_file_sing( mapped, size );    

    if ( munmap(mapped, size) == -1 )
    {
        close( fd );
        perror( "Error! un_mmapping file! " );
        exit( EXIT_FAILURE );
    }

    close( fd );
    
    return traj;
}

Trajectory Construct::parse_file_wz( char *mapped, size_t size ) 
{

    // get lines of the file
    int num = count_line( mapped, size );

    printf(" line number: %d\n", num);

    Trajectory traj( num );

    char attrs[30];
    char *head, *tail;
    head = mapped;
    tail = head;
    int idx = 0;

    while ( idx < num )
    {
        // go to seperator ',' for tid
        while ( *tail != ',' ) tail++;
        
        // copy tid between head and tail
        strncpy( attrs, head, tail-head );
        head = tail+1;
        tail = head;    // jump over ','
        
        //printf( "%d\t", atoi(attrs) );
        traj.set_id( atoi( attrs ) );
 
        // go to seperator ',' for longitude
        while ( *tail != ',' ) tail++;
        //printf("%ld\n", tail-head);
        strncpy( attrs, head, tail-head );
        head = tail+1;
        tail = head;
        traj.add_lon( atof(attrs), idx );

        // go to seperator ',' for latitude
        while ( *tail != ',' ) tail++;
        strncpy( attrs, head, tail-head );
        head = tail+1;
        tail = head;
        traj.add_lat( atof(attrs), idx );

        // go to seperator ',' for time
        while ( *tail != ',' ) tail++;
        strncpy( attrs, head, tail-head );
        attrs[tail-head+1] = '\0';
        head = tail+1;
        tail = head;

        printf("time: %s, line: %d, ts: %ld\n", attrs, idx, atots(attrs));
        traj.add_ts( atots(attrs), idx );
       
        // go to '\n' for end of a line
        while ( *tail != '\n' ) tail++;
        head = tail+1;
        tail = head;
        idx ++;
    }

    return traj;   
}

/* parse files of MSRA taxi data */
Trajectory Construct::parse_file_msra( char *mapped, size_t size ) 
{

    // get lines of the file
    int num = count_line( mapped, size );

    printf(" line number: %d\n", num);

    Trajectory traj( num );

    char attrs[30];
    char *head, *tail;
    head = mapped;
    tail = head;
    int idx = 0;

    while ( idx < num )
    {
        // go to seperator ',' for tid
        while ( *tail != ',' ) tail++;
        
        // copy tid between head and tail
        strncpy( attrs, head, tail-head );
        //printf("%ld\n", tail-head);
        attrs[tail-head] = '\0';
        head = tail+1;
        tail = head;    // jump over ','
        
        //printf( "%s\n", attrs );
        traj.set_id( atoi( attrs ) );
 
        // go to seperator ',' for time
        while ( *tail != ',' ) tail++;
        strncpy( attrs, head, tail-head );
        attrs[tail-head+1] = '\0';
        head = tail+1;
        tail = head;

        //printf("time: %s, line: %d, ts: %ld\n", attrs, idx, atots(attrs));
        traj.add_ts( atots(attrs), idx );

        // go to seperator ',' for longitude
        while ( *tail != ',' ) tail++;
        //printf("%ld\n", tail-head);
        strncpy( attrs, head, tail-head );
        attrs[tail-head+1] = '\0';
        head = tail+1;
        tail = head;
        traj.add_lon( atof(attrs), idx );

        // go to seperator ',' for latitude
        while ( *tail != '\n' ) tail++;
        strncpy( attrs, head, tail-head );
        attrs[tail-head+1] = '\0';
        head = tail+1;
        tail = head;
 
        //printf("%f\t", atof(attrs));
        traj.add_lat( atof(attrs), idx );

        idx ++;
    }

    return traj;   
}

/* parse files of Singapore taxi data */
Trajectory Construct::parse_file_sing( char *mapped, size_t size ) 
{

    // get lines of the file
    int num = count_line( mapped, size );

    printf(" singarpore taxi line number: %d\n", num );

    Trajectory traj(num);

    char attrs[30];
    char *head, *tail;
    head = mapped;
    tail = head;
    int idx = 1;

    // jump the header
    while (*tail != '\n') tail++;

    while ( idx < num )
    {
        // go to seperator ',' for tid
        while ( *tail != ',' ) tail++;
        tail++;
        head = tail;
        
        // go to seperator ',' for time
        while ( *tail != ',' ) tail++;
        strncpy( attrs, head, tail-head );
        attrs[tail-head+1] = '\0';
        head = tail+1;
        tail = head;

        printf("time: %s, line: %d, ts: %ld\n", attrs, idx, atots_sg(attrs));
        traj.add_ts( atots_sg(attrs), idx );

        // copy tid between head atots_sgand tail
        while ( *tail != ',' ) tail++;
        strncpy( attrs, head, tail-head );
        //printf("%ld\n", tail-head);
        attrs[tail-head] = '\0';
        head = tail+1;
        tail = head;    // jump over ','
        
        // printf( "%s\n", attrs );
        traj.set_id(atoi( attrs ));

        // go to seperator ',' for longitude
        while ( *tail != ',' ) tail++;
        //printf("%ld\n", tail-head);
        strncpy( attrs, head, tail-head );
        attrs[tail-head+1] = '\0';
        head = tail+1;
        tail = head;
        traj.add_lon( atof(attrs), idx );

        // go to seperator ',' for latitude
        while ( *tail != ',' ) tail++;
        strncpy( attrs, head, tail-head );
        attrs[tail-head+1] = '\0';
        head = tail+1;
        tail = head;
 
        //printf("%f\t", atof(attrs));
        traj.add_lat( atof(attrs), idx );

        // go to '\n' for end of a line, ommitting the other attributes
        while ( *tail != '\n' ) tail++;
        head = tail+1;
        tail = head;
        idx ++;
    }

    return traj;   
}

int Construct::find_files( const char* folder )
{
    DIR * dir;
    struct dirent * ent;


    int level = 0;

    // parse folder like "11_1100_n" to get level
    // const char *l, *t;
    // if ( (l = strrchr(folder, '/')) && (t = strchr(folder, '_')) != NULL ) 
    // {
    //     int n = t - l;   // level's length in string
    //     char ss[5];
    //     assert(n < 4);
    //     strncpy(ss, l+1, n-1);
    //     ss[n] = '\0';
    //     level = atoi(ss);

    //     printf("----------level: %d\n", level);
    // }

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
                char file_name[100];
                sprintf( file_name, "%s/%s", folder, ent->d_name );
                strcpy( file_array[num_file++], file_name );
                //printf("file number: %d, %s\n", num_file, file_name);
             }
        }
        
        closedir( dir );
    }
    else
    {
        perror( "read file error!" );
        return -1;
    }

    return level;
} 

void Construct::save_cells(vector<Item> &cells, char *path)
{
    FILE *file = fopen(path, "w");

    printf("cell size: %lu\n", cells.size());

    //! using calloc, else report memory overlap error!!
    char *buf = (char *) calloc(cells.size() * 90, sizeof(char));
    uint64 last_cell_id = 0;
    for (int i = 0; i < cells.size(); i++)
    {    
        char line[90];
        // format a line
        uint64 current_id = cells[i].cell_id;
                          
        // if (current_id == 3886690940026355712llu && (last_cell_id - current_id) == 0)
        //     printf("===================%llu\n", last_cell_id);
        if ((last_cell_id - current_id) != 0) 
        {
            snprintf(line, sizeof(line), "%llu,%d,%ld\n", 
                        current_id, cells[i].tid, cells[i].ts);
            strcat(buf, line);
        }

        last_cell_id = current_id;
    }

    fprintf(file, "%s", buf);
    if (cells.size() > 0)
        cells.clear();
    fclose(file);
    free(buf);
}

int Construct::count_line( char * mapped, size_t size )
{
    int line = 0;
    register int ss = 0;

    while ( ss < size )
    {
        if ( *mapped == '\n' )  line++;
        mapped++;
        ss++;
    }

    return line;
}

long Construct::atots( char *s )
{
    int year, month, day, hour, min, second;
    long int acc;
    int z = 0, c;

    for ( year = 0; z < 4; )
    {
        c = (unsigned char) *s++;

        if ( c != '-')
        {
            c-= '0';
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
        month *=10;
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

    // evaluating time in second
    acc = (year - 1970) * 365 * 24 * 3600 + (cnt+day-1) * 24 * 3600 + (hour-8) * 3600 + min * 60 + second;
    int feb = ((year%4==0&&year%100!=0) || (year%400==0)) ? 29 * 24 * 3600 : 28 * 24 * 3600;
    
    switch (month)
    {
        case 1: break;
        case 2: acc += 31 * 24 * 3600;      // value of January + 31 days
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

    return acc;
}


long Construct::atots_sg( char *s )
{
    int year, month, day, hour, min, second;
    long int acc;
    int z = 0, c;

    for (day=0; z < 2; )
    {
        c = (unsigned char) *s++;
        c -= '0';
        day *= 10;
        day += c;
        z++;
    }

    // jump '/'
    c = *s++;
    z = 0;
    
    for (month = 0; z < 2; )
    {
        c = (unsigned char) *s++;
        c -= '0';
        month *=10;
        month += c;
        z++;
    }

    // jump '/'
    c = *s++;
    z = 0;
    
    for ( year = 0; z < 4; )
    {
        c = (unsigned char) *s++;

        if ( c != '-')
        {
            c-= '0';
            year *= 10;
            year += c;
        }
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

    // evaluating time in second
    acc = (year - 1970) * 365 * 24 * 3600 + (cnt+day-1) * 24 * 3600 + (hour-8) * 3600 + min * 60 + second;
    int feb = ((year%4==0&&year%100!=0) || (year%400==0)) ? 29 * 24 * 3600 : 28 * 24 * 3600;
    
    switch (month)
    {
        case 1: break;
        case 2: acc += 31 * 24 * 3600;      // value of January + 31 days
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

    return acc;
}


