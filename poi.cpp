#include <stdio.h>

#include "poi.h"
#include "s2/s2latlng.h"
#include "s2/s2cell.h"

void S2_POI::transform(POI *poi, int l)
{
    // get s2 latitude and longitude of poi
    S2LatLng lat_lng = S2LatLng::FromDegrees( poi->latitude, poi->longitude );

    // get s2 cell id of level l
    S2CellId cid = S2CellId::FromLatLng( lat_lng );
    S2CellId cid_l = cid.parent( l );

    set_cell_id( cid_l.id() );

    // simply deal with keywords
    set_keywords( poi->name );
    set_type( poi->type );

}

void POI_Data::batch_transform(int l)
{
    // s2_poi_map.clear();
    std::map<const std::string, std::vector<POI> >::iterator it, end;
    for (it = poi_map.begin(), end = poi_map.end(); it != end; ++it)
    {
        std::vector<POI> vec = it->second;

        std::vector<POI>::iterator vit, vend;
        for (vit = vec.begin(), vend = vec.end(); vit != vend; ++vit)
        {
            S2_POI s2_poi;
            s2_poi.transform(&*vit, l);

            std::map<const std::string, std::vector<uint64> >::iterator it = s2_poi_map.find((const char*) s2_poi.type);

            if (it != s2_poi_map.end()) 
            {
                std::vector<uint64> vec = it->second;
                // the cell id is not existing in vector
                if (std::find(vec.begin(), vec.end(), s2_poi.cell_id) == vec.end())
                    vec.push_back(s2_poi.cell_id);
                it->second = vec;   // very important!!! if not, the map will not change.
            }
            else 
            {
                // not found, add pair in the map
                std::vector<uint64> vec;
                vec.push_back( s2_poi.cell_id ); 
                s2_poi_map.insert( std::map<const std::string, std::vector<uint64>, cmp_str>::value_type(s2_poi.type, vec) );                        
            }
        }
    }

}


void POI_Data::read_poi(char *filename)
{
    char line[1000];
    char name[100], type[100];
    float lon, lat;

    FILE *fp;


    if ( (fp = fopen(filename, "r")) == NULL )
    {
        printf("No Such File\n");
        exit(1);
    } 

    int count = 0;
    while ( fgets(line, sizeof(line), fp) )
    {
        int ret = sscanf(line, "%[^,],%[^,],%f,%f", name, type, &lon, &lat);

        // printf("-----%f\n", lon);
        if (ret == 4)
        {
            POI *poi = (POI*) malloc(sizeof(POI));
            if (strlen(name) > 0)
            {
                poi->name = (char*) malloc( strlen(name) );
                memcpy( poi->name, name, strlen(name) );
            }
            else
                poi->name = NULL;
            
            if (strlen(type) > 0)
            {
                poi->type = (char*) malloc( strlen(type) );
                memcpy( poi->type, type, strlen(type) );

                if ( strcmp("科教文化服务;传媒机构;出版社", poi->type) == 0 )
                    count++;
                      // printf("----%s\n", poi->type);
            }
            else
                poi->type = NULL;

            poi->longitude = lon;
            poi->latitude = lat;
            

            if ( strcmp(name, "龙台上") == 0)
                printf("name: %s, lon: %f\n", name, lon);

            std::string type (poi->type);
            std::map<const std::string, std::vector<POI> >::iterator it = poi_map.find(type);

            if (it != poi_map.end())    //found
            {
                it->second.push_back(*poi); 

    // std::string str ("汽车服务;加油站;中国石化");
    //             if (str.compare(std::string(poi->type)) == 0) 
    //                 printf("vec size: %lu\n", poi_map[str].size());
            }
            else
            {
                vector<POI> vec;
                vec.push_back(*poi);
                poi_map.insert(std::pair<const std::string, std::vector<POI> >(poi->type, vec));
            }
        } 

    }

    printf("line number: %d\n", count);


    fclose(fp);
}

// struct Find_POI
// {

//     Find_POI(POI p, std::vector<POI> *v):P(p), vec(v) {}

//     void operator()(POI p) 
//     {
//         // if ( strcmp("风景名胜;风景名胜相关;旅游景点", p.type) == 0 )
//         //     printf("%s, ----%s\n", P.type, p.type);
        
//         if ( strcmp(p.type, P.type) == 0 )
//         {
//             vec->push_back( p.get_cell_id() );
//             // printf("cell id in poi: %llu\n", p.get_cell_id());
//         }
//     }

//     POI P;
//     std::vector<uint64> *vec;
// };

std::vector<uint64> POI_Data::search_s2_poi_by_type(char *type)
{
    // find many pois
    // Find_POI fp(poi, &cid_vec);
    // std::for_each(
    //     s2_poi_vec.begin(), 
    //     s2_poi_vec.end(), 
    //     fp);

    return s2_poi_map[poi.type];
}

std::vector<POI> POI_Data::search_poi_by_type(char *tp)
{
    return poi_map[tp];
}

POI POI_Data::search_poi_by_type_keyword(char *type, char *keyword)
{
    std::vector<POI> vec = search_by_type(type);

    std::vector<POI>::iterator ite = std::find_if(vec.begin(), vec.end(), [] (const POI p) { return strcmp(p.name, keyword) == 0;})

    if (ite != vec.end())
        return *ite;
    else
        return POI();
}

std::map<const std::string, std::vector<POI> > get_all_poi_types()
{
    return poi_map;
}