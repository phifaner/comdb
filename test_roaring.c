#include <iostream>
#include <assert.h>

//#include "roaring/roaring.h"
#include "roaring.hh"
#include "roaring.c"

/*int main() {
    // create a new empty bitmap
    roaring_bitmap_t *r1 = roaring_bitmap_create();
    // add values
    for (uint32_t i = 100; i < 1000; i++) roaring_bitmap_add(r1, i);

    // compute how many bits there are
    printf("cardinality = %d\n", (int) roaring_bitmap_get_cardinality(r1));

    // if your bitmaps have long runs, you can compress them by calling
    // run_optimize
    uint32_t expectedsizebasic = roaring_bitmap_portable_size_in_bytes(r1);
    roaring_bitmap_run_optimize(r1);
    uint32_t expectedsizerun= roaring_bitmap_portable_size_in_bytes(r1);
    printf("size before optimize %d bytes, and after %d bytes\n", 
            expectedsizebasic, expectedsizerun);
    //roaring_bitmap_printf(r1);

    // create a new bitmap containing the values {1,2,3,5,6}
    roaring_bitmap_t *r2 = roaring_bitmap_of(5, 1, 2, 3, 5, 6);
    roaring_bitmap_printf(r2);

    // we can also create a bitmap from a pointer to 32-bit integers
    uint32_t somevalues[] = {2, 3, 4};
    roaring_bitmap_t *r3 = roaring_bitmap_of_ptr(3, somevalues);
    
    // we can also go in reverse and go from arrays to bitmaps
    uint64_t card1 = roaring_bitmap_get_cardinality(r1);
    uint32_t *arr1 = (uint32_t *) malloc(card1 * sizeof(uint32_t));
    assert(arr1  != NULL);
    roaring_bitmap_to_uint32_array(r1, arr1);
    roaring_bitmap_t *r1f = roaring_bitmap_of_ptr(card1, arr1);
    free(arr1);
    assert(roaring_bitmap_equals(r1, r1f));  // what we recover is equal
    roaring_bitmap_free(r1f);

    // we can copy and compare bitmaps
    roaring_bitmap_t *z = roaring_bitmap_copy(r3);
    assert(roaring_bitmap_equals(r3, z));  // what we recover is equal
    roaring_bitmap_free(z);
    
    // we can compute union two-by-two
    roaring_bitmap_t *r1_2_3 = roaring_bitmap_or(r1, r2);
    roaring_bitmap_or_inplace(r1_2_3, r3);

    // we can compute a big union
    const roaring_bitmap_t *allmybitmaps[] = {r1, r2, r3};
    roaring_bitmap_t *bigunion = roaring_bitmap_or_many(3, allmybitmaps);
    assert(
     roaring_bitmap_equals(r1_2_3, bigunion));  // what we recover is equal
    // can also do the big union with a heap
    roaring_bitmap_t *bigunionheap = roaring_bitmap_or_many_heap(3, allmybitmaps);
    assert(roaring_bitmap_equals(r1_2_3, bigunionheap));

    roaring_bitmap_free(r1_2_3);
    roaring_bitmap_free(bigunion);
    roaring_bitmap_free(bigunionheap);

    // we can compute intersection two-by-two
    roaring_bitmap_t *i1_2 = roaring_bitmap_and(r1, r2);
    roaring_bitmap_free(i1_2);

    roaring_bitmap_free(r1);
    roaring_bitmap_free(r2);
    roaring_bitmap_free(r3);
    return 0;
}
*/

int main() {
  Roaring r1;
  for (uint32_t i = 100; i < 1000; i++) {
    r1.add(i);
  }
  std::cout << "cardinality = " << r1.cardinality() << std::endl;

  Roaring64Map r2;
  for (uint64_t i = 18000000000000000100ull; i < 18000000000000001000ull; i++) {
    r2.add(i);
  }
  std::cout << "cardinality = " << r2.cardinality() << std::endl;
  return 0;
}