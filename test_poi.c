#include <stdio.h>

#include "poi.h"

void test_transform()
{
    char *name = "pansir panjang hotel";
    char *type = "hotel";
    float latitude = 28.233;
    float longitude = 121.345;
    POI poi;
    poi.name = name;
    poi.type = type;
    poi.longitude = longitude;
    poi.latitude = latitude;

    S2_POI spoi;
    spoi.transform(&poi, 12);

    printf("name: %s, cell id: %llu\n", spoi.keywords, spoi.cell_id);
}

void test_read_poi()
{
    POI_Data pd;
    pd.read_poi("./beijing.csv");
}

void test_batch_transform()
{
    POI_Data pd;
    pd.read_poi("./beijing.csv");
    
    pd.batch_transform(12);

    // printf("size: %lu\n", pd.s2_poi_vec.size());    
}

void test_search_s2_poi_by_type()
{
    POI_Data pd;
    pd.read_poi("beijing.csv");

    pd.batch_transform(12);
    
    //char *kw = "小西天";
    //char *kw = "renmin university";
    char *kw = "";
    //char *tp = "购物服务;专卖店;音像店";
    char *tp = "风景名胜;风景名胜;风景名胜";
    // char *tp = "科教文化服务;传媒机构;出版社";
    std::vector<uint64> vec = pd.search_s2_poi_by_type(tp);

    printf("vec size: %lu\n", vec.size());
    
}

void test_search_poi_by_type()
{
    POI_Data pd;
    pd.read_poi("beijing.csv");

    // pd.batch_transform(12);

    //char *kw = "小西天";
    //char *kw = "renmin university";
    char *kw = "";
    //char *tp = "购物服务;专卖店;音像店";
    char *tp = "汽车服务;加油站;中国石化";
    // char *tp = "科教文化服务;传媒机构;出版社";
    std::vector<POI> vec = pd.search_poi_by_type(tp);

    printf("latitude: %lf\n", vec[0].latitude);
    
}

#include <locale.h>
#include <stdio.h>
#include <wchar.h>
#include <string.h>
#include <stdlib.h>
//int
//main(int argc, char *argv[])
//{
    //FILE   *infile = fopen(argv[1], "r");
   // wchar_t test[2] = L"\u4E2A";
   // setlocale(LC_ALL, "");
   // printf("%ls\n", test);  //test
   // wcscpy(test, L"\u4F60");        //test
   // printf("%ls\n", test);  //test
   // for (int i = 0; i < 5; i++) {
     //   fscanf(infile, "%1ls", test);
       // printf("%ls\n", test);
    //}
   
 //   return 0;
//}

int main()
{
    //test_transform();
    //test_read_poi();
    // test_batch_transform();
    test_search_poi();
}
