#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#include "hash_table.h"
#include "file_index.h"

unsigned long hash(unsigned char *str)
{
    unsigned long hash = 5381UL;
    int c;

    while (c = *str++)
        hash = ((hash << 5) + hash) + c; /* hash * 33 + c */

    return hash;
}

unsigned long hash_file(char *tid, char *date)
{
    char ss[50];
    strcpy(ss, date);
    strcat(ss, tid);
    
    return hash((unsigned char *)ss);
}

void read_index(const char* filename)
{
    unsigned long id;
    char *path = (char *)malloc(60);
    FILE * file;
    char * line = NULL;
    size_t len = 0;
    ssize_t read;

    if ((file = fopen(filename, "r")) == NULL)
    {
        printf("Error! opening file\n");
        exit(1);
    }
    
    while (fscanf(file, "%lu,%s", &id, path) != EOF) 
    {
        insert(id, path);
        path = (char *)malloc(60);      /* !need to allocate memory! */
    }
    
   /* while ((read = getline(&line, &len, file)) != -1) 
    { 
        //printf("Data in the file: %s \n", line);
        
        // parse line
        memcpy(&id, line, sizeof(unsigned long));

        //printf("id: %lu\n", id);

        char * p = strchr(line, ',');

        strcpy(path, p+1);

        //printf("path: %s\n", path);

        // insert into a hash table, hash table is defined in hash_table.h
        insert(table, id, path);    
    }*/
   
    printf("Data insert into file close \n");

    fclose(file);

}

const char * get_path(char * tid, char * date)
{
    /* get hash code */
    unsigned long code =  hash_file(tid, date);
    
    /* get file path from hash table */
    struct item * t = get_by_key(code);

    return t->value;
}


/*int main()
{
    char *date = "20140101";
    char *tid = "C02668";

    unsigned long v =  hash_file(tid, date);
    
    printf("hash value: %lu\n", v);

    //struct item ** table = create();

    read_index("index.txt");
    
    //delete(17756674295831246169UL);

    //struct item *p_item = search(17756674167933085911UL);
    struct item *p_item = search(17756674295831246169UL);

    //delete(p_item);

    //p_item = search(17756674295831246169UL);

    if (p_item != NULL)
        printf("Element found: %s\n", p_item->value);
    else
        printf("Element not found\n");
}
*/
