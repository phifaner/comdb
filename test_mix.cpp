#include "mix.h"
#include "roaring/roaring.h"

void test_hindex_create()
{
    Item i1 = {3, 5, 11212};
    Item i2 = {200, 5, 3234};
    Item i3 = {1098882, 5, 97872};
    Item i4 = {83, 5, 123};

    std::vector<Item> items;
    items.push_back(i1);
    items.push_back(i2);
    items.push_back(i3);
    items.push_back(i4);

    //std::vector<uint32_t> times;
    //times.push_back(11112);
    //times.push_back(3234);
    //times.push_back(97872);
    //times.push_back(123);

    // H_Index * hi = hindex_create_bitmap(items.data(), 4, 5);
    // roaring_bitmap_printf(hi->cell_bitmap);

    // assert(roaring_bitmap_contains(hi->cell_bitmap, 200));
}

void test_path()
{
    int level;
    char *folder = "/home/wangfei/11_06_n";

    const char *l, *t;
    if ( (l = strrchr(folder, '/')) && (t = strchr(folder, '_')) != NULL ) 
    {
        int n = t - l;   // level's length in string

        char ss[5];
        assert(n < 4);
        strncpy(ss, l+1, n-1);
        ss[n] = '\0';
                printf("%s\n", ss);
        level = atoi(ss);

        printf("----------level: %d\n", level);
    }
}

void test_all_hindex_create()
{
    Item i1 = {3, 5, 11212};
    Item i2 = {200, 5, 3234};
    Item i3 = {1098882, 5, 97872};
    Item i4 = {83, 5, 123};

    std::vector<Item> items;
    items.push_back(i1);
    items.push_back(i2);
    items.push_back(i3);
    items.push_back(i4);

    Item i1_1 = {3, 6, 11212};
    Item i2_1 = {200, 6, 3234};
    Item i3_1 = {1098882, 6, 97872};
    Item i4_1 = {83, 6, 123};

    std::vector<Item> items_1;
    items_1.push_back(i1_1);
    items_1.push_back(i2_1);
    items_1.push_back(i3_1);
    items_1.push_back(i4_1);

    std::vector<vector<Item> > traj_vec;
    traj_vec.push_back(items);
    traj_vec.push_back(items_1);

    assert(traj_vec.size() == 2);
    assert(traj_vec[1][1].tid == 6);
}

int main()
{
    // test_hindex_create();
    // test_all_hindex_create();
    test_path();
}
