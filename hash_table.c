#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "hash_table.h"

struct item ** create()
{
    struct item ** hash_array = (struct item **)malloc(SIZE * sizeof(struct item *));
    if (hash_array == 0) return 0;

    return hash_array;   
}


struct item * get_by_key(unsigned long key)
{
    // get the hash code
    int code = hash_code_key(key);

    // move in array until an empty slot
    while (table[code] != NULL)
    {
        if (table[code]->key == key)
            return table[code];

        // go to the next slot
        ++code;

        // wrap around the table
       code %= SIZE;
    }

    return NULL;
}

int insert(unsigned long key, const char * value)
{
    struct item *p_item = (struct item *)malloc(sizeof(struct item));
    if (p_item == 0) return 0;

    p_item->key = key;
    p_item->value = value;

    //if (key == 17756674295831246169)
    //    printf("Element inserted: %p\n", (void*)p_item->value);

    // get the hash code
    int code = hash_code_key(key);

    // move in array until an empty slot or delete cell
    while (table[code] != NULL && table[code]->key != -1)
    {
        // go to the next slot
        ++code;

        // wrap around the table
        code %= SIZE;
    }

    table[code] = p_item;

    return 1;
}

struct item * deletee(struct item * _item)
{
    if (table == NULL) return NULL;

    struct item * dummy_item = (struct item *)malloc(sizeof(struct item));
    dummy_item->key = -1;
    dummy_item->value = "";

    unsigned long key = _item->key;

    // get the hash code
    int code = hash_code_key(key);

    // move in array until an empty slot
    while (table[code] != NULL)
    {
        if (table[code]->key == key)
        {
            struct item* temp = table[code];

            // assign a dummy item at the deleted position
            table[code] = dummy_item;
            
            printf("delete element: %d\n", code);

            return temp;
        
        }

        ++code;

        code %= SIZE;
    }

    return NULL;
}

void display(struct item ** table) 
{
    int i = 0;

    for (i = 0; i < 20; i++)
    {
        if (table[i] != NULL)
            printf(" (%lu,%s) ", table[i]->key, table[i]->value);
        else
            printf(" ~~ ");
    }

    printf("\n");
}

/*int main()
{
    dummy_item = malloc(sizeof(struct item));
    dummy_item->key = -1;
    dummy_item->value = "";

    insert(1, "20");
    insert(2, "70");
    insert(42, "80");
    insert(4, "70");
    insert(12, "20");
    insert(14, "70");
    insert(17, "20");
    insert(13, "70");
    insert(37, "/home/wangfei/201401TRK/TRK20140131/CT6/CT6128.txt");
    //insert(2, "70");
    
    display();

    struct item *_item = search(37);
    if (_item != NULL)
    {
        printf("Element found: %s\n", _item->value);
    } 
    else
    {
        printf("Element not found\n");
    }

    delete(_item);
    _item = search(37);

    if (_item != NULL)
    {
        printf("Element found: %s\n", _item->value);
    } 
    else
    {
        printf("Element not found\n");
    }
}*/

