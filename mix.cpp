#include "mix.h"
#include "s2/s2cellid.h"
#include "roaring.c"

#include <math.h>

/*H_Index * hindex_create_old(Item * T, size_t len, unsigned level)
{

    // test if cells < 2^32, only 32-bit bitmap
    assert(len < 2^32);
    assert(T != NULL);
       
    H_Index *p_index = (H_Index*) malloc(sizeof(H_Index));
    p_index->length = len;

    unsigned long *ts_array = (unsigned long*) malloc(len * sizeof(unsigned long));

    // assign values for bitmap   
    register size_t size;
    roaring_bitmap_t *r = roaring_bitmap_create();
    for (size = 0; size != len; ++size) 
    {
        roaring_bitmap_add(r, T[size].cell_id);
        ts_array[size] = T[size].ts;   
    }  

    p_index->tid = T[0].tid;
    p_index->level = level;
    p_index->cell_bitmap = r;
    p_index->s_array = ts_array;
 
    return p_index;
}
*/
void HMix::hindex_create(H_Points points, unsigned level)
{
    register uint64_t *px = (uint64_t*) points.X;
    register unsigned long *py = points.Y;
    register int *pid = points.ID;
    register size_t len = points.length;

    assert(px != NULL);
    assert(len > 0);



    // if two consecutive ids are not equal
    // then, it means starting a new trajectory
    register int last_id = pid[0];
    register int current_id = pid[0];
    register size_t num = 0;
    
    //S2CellId send = S2CellId::Begin(level-1);
    //S2CellId sbegin= S2CellId::Begin(level);
    //size_t cell_size = 6 * pow(2, level) * pow(2, level);
    // uint32_t cell_size = pow(2, 32) - 1;

    // printf("cell size %u\n", cell_size);

    //TODO release memory
    // unsigned long *ts_array = 
    //     (unsigned long*) calloc(cell_size, sizeof(unsigned long));
    while ( num < len )
    {
        // start a new trajectory index
        H_Index p_index ;
        //(H_Index*) malloc(sizeof(H_Index));
        std::map<uint64, unsigned long> s_map;

        Roaring64Map *r = new Roaring64Map();
        size_t max_idx = 0;
   
        // if equal, go to the next point
        while (last_id != current_id) 
        {
            last_id = current_id;
            current_id = pid[++num];
        }

        while (last_id == current_id)
        {
            // printf("num: %lu, time: %lu\n", num, py[num]);

            // assign value for bitmap and inverted list
            r->add(px[num]);
            
            //if (px[num] == 3886682874077773824ll)      
               // printf("high bytes index: -------- %lu, %lu\n", px[num], py[num]);

            //  only take high bytes of cell id
            // ts_array[highBytes(px[num])] = py[num]; 
            std::map<uint64, unsigned long>::iterator it = s_map.find(px[num]);

            if (it == s_map.end())
                s_map.insert( make_pair(px[num], py[num]) );

            last_id = current_id;
            current_id = pid[++num];
            // if ( max_idx < highBytes(px[num]) ) max_idx = highBytes(px[num]);
            
            // assert (max_idx < cell_size);      
        }

        // unsigned long *ts_array_new = (unsigned long*) calloc(max_idx, sizeof(unsigned long));
        // memcpy(ts_array_new, ts_array, max_idx * sizeof(unsigned long));

        // put the current trajectory index into vector 
        p_index.tid = last_id;
        p_index.level = level;
        p_index.cell_bitmap = r;
        p_index.s_map = s_map;
        // p_index->length = max_idx;

        // recycle memory
        //free(ts_array);
        hmap.insert( make_pair(last_id, p_index) );
    }

    // free(ts_array);

}


