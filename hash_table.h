#ifndef _HASH_TABLE_H
#define _HASH_TABLE_H

#define SIZE 120000     /* table size, file number */

struct item
{
    unsigned long       key;
    const char  *       value;
};

static struct item * table[SIZE];
static struct item * dummy_item;

static inline int hash_code_key(unsigned long key)
{
    return key % SIZE;
}

extern "C" struct item * get_by_key(unsigned long);

extern "C" int insert(unsigned long key, const char* value);

extern "C" struct item * deletee(struct item *);
#endif


