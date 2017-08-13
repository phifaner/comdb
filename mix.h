#include "construct.h"
#include "roaring.hh"

#include <vector>

/**
 * we only support 32-bit bitmap, therefore the cells' number of one level is less than 2^32
 * TODO extend to 64-bit bitmap
 */

using namespace std;

struct H_Index
{
    // trajectory id
    unsigned tid;

    // level of s2
    unsigned level;

    // bitmap for a trajectory's passing tmapcells
    Roaring64Map *cell_bitmap;

    // inverted list of timestamp associated with the bitmap
    // unsigned long *s_array;
    // unsigned long *e_array;
    std::map<uint64, unsigned long> s_map;
    std::map<uint64, unsigned long> e_map;
    // length of trajectory's cells
    size_t length;          
};

struct H_Points
{
    uint64          *X;
    unsigned long   *Y;
    int             *ID;
    size_t          length;
};

class HMix
{
public:
    // create the index for a trajectory of level l
    H_Index * hindex_create(Item* traj, size_t len, unsigned level);

    // create indexes for all trajectories
    void hindex_create(H_Points points, unsigned level);

    // below are querying on the index

    //size_t estimate(Roaring64Map *R, Roaring64Map L);

    // R: a bitmap of query cells
    // T: time span 
    vector<unsigned> query(Roaring64Map *R, unsigned long *T);

    // return index
    map<int, H_Index> get_hmap() { return hmap; }

    void set_hmap(map<int, H_Index> map) { hmap = map; }

    // map<int, D_Index> get_dmap() { return dmap; }

    // void set_dmap(map<int, D_Index> map) { dmap = map; }

    // map<int, H_Index> get_hmap_12() { return hmap_12; }

private:

    inline uint32_t highBytes(const uint64_t in) { return uint32_t(in >> 32); }

    inline uint32_t lowBytes(const uint64_t in) { return uint32_t(in); }

    // store mix indexes
    // vector<H_Index *> hmix;
    map<int, H_Index> hmap;
    // map<int, D_Index> dmap;
    // map<int, H_Index> hmap_12;
};

