#include "construct.h"

#include <stdio.h>

void test_intersect()
{
    double x[2] = {1, 5};
    double y[2] = {1, 3};
    double ll[2] = {2, 1};
    double ur[2] = {4, 3};
    
    Construct c;
    if ( c.intersect( x, y, ll, ur ) )
        printf( "......Intersection...\n" );
}

void test_assign_line()
{
    double x[2] = {120.1, 120.5};
    double y[2] = {28.1, 28.3};
    unsigned long ts[2] = {5000, 7000};

    Construct c;
    std::vector<Item> cells = c.assign_line( 100, x, y, ts, 10 );
    for (int i = 0; i < cells.size(); i++ )
        printf( "cell id: %llu\n", cells[i].cell_id );
}

void test_assign()
{
    const char *file = "/home/wangfei/201401TRK/TRK20140101/C02/C02668.txt";
    
    Construct c;
    Trajectory p_traj = c.load_file( file );
    std::vector<Item> cells = c.assign( &p_traj, 15 );
    for (int i = 0; i < cells.size(); i++)
        printf( "cell id: %llu\n", cells[i].cell_id );
}

void test_load_file()
{
    const char *file = "/home/wangfei/06/1129.txt";

    Construct c;
    c.load_file( file );
}

void test_find_files()
{
    const char *path = "/home/wangfei/201401TRK";
    Construct c;
    c.find_files(path);
    //printf("number of files: %d\n", c.num_file);
}

void test_save_cells()
{
    //const char *path = "/home/wangfei/201401TRK/TRK20140101/C02/C02668.txt";
    // const char *path = "/home/wangfei/06/443.txt";
    // Construct c;
    // Trajectory traj = c.load_file(path);
    // std::vector<Item> cells = c.assign(&traj, 11);
    // c.save_cells(cells, "/home/wangfei/06/443_n.txt");
    
    const char *path = "/home/wangfei/06";

    Construct c;
    c.find_files(path);
    int level = 13;
    
    char ** files = c.get_file_array();
    int num_file = c.get_file_num();

    for (int i = 0; i != num_file; i++)
    {
        printf("start to parse file : %s\n", files[i]);
        Trajectory p_traj = c.load_file(files[i]);
        std::vector<Item> cells = c.assign(&p_traj, level);
        
        char *ss = strtok(files[i], ".");
        //strncpy(ss, files[i], strlen(files[i])-5);
        //printf("ss: %s\n", ss);
        sprintf( ss, "%s_n.txt", ss);
        printf("---ss: %s\n", ss);
        c.save_cells(cells, ss);
    }
    
    //c.save_cells(cells, "/home/wangfei/201401TRK/TRK20140101/C02/C02668_n.txt");
}

int main()
{
    printf("----------\n");
    //test_intersect();
    //test_assign_line();
    //test_load_file();
    //test_find_files();
    //test_assign();
    test_save_cells();
    printf("end\n");
}
