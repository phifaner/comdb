#include "s2_index.h"
#include "poi.h"
#include "roaring/roaring.h"
#include <ctime>

std::vector<S2_POI> set_poi()
{
    // char *tp_1 = "餐饮服务;中餐厅;中餐厅";
    // char *tp_2 = "生活服务;美容美发店;美容美发店";
    // char *tp_3 = "公司企业;公司;公司";
    // char *tp_4 = "风景名胜;风景名胜相关;旅游景点";
    // char *tp_5 = "科教文化服务;传媒机构;出版社";
    // char *tp_6 = "科教文化服务;学校;高等院校";
    // char *tp_7 = "生活服务;中介机构;中介机构";
    // char *tp_8 = "购物服务;综合市场;农副产品市场";
    // char *tp_9 = "餐饮服务;外国餐厅;韩国料理";
    // char *tp_10 = "体育休闲服务;娱乐场所;游戏厅";
    // char *tp_11 = "科教文化服务;学校;小学";
    // char *tp_12 = "汽车维修;汽车综合维修;汽车综合维修";
    // char *tp_13 = "购物服务;便民商店/便利店;便民商店/便利店";
    // char *tp_14 = "住宿服务;宾馆酒店;宾馆酒店";
    // char *tp_15 = "购物服务;专卖店;专营店";
    // char *tp_16 = "公司企业;公司;网络科技";
    // char *tp_17 = "购物服务;专卖店;眼镜店";
    // char *tp_18 = "政府机构及社会团体;公检法机构;社会治安机构";
    // char *tp_19 = "商务住宅;住宅区;住宅小区";
    // char *tp_20 = "医疗保健服务;急救中心;急救中心";

    char *tp_1 = "科教文化服务;美术馆;美术馆";
    char *tp_2 = "金融保险服务;银行;汇丰银行";
    char *tp_3 = "公司企业;公司;矿产公司";
    char *tp_4 = "体育休闲服务;影剧院;剧场";
    char *tp_5 = "商务住宅;楼宇;工业大厦建筑物";
    char *tp_6 = "餐饮服务;外国餐厅;美式风味";
    char *tp_7 = "购物服务;超级市场;北京华联";
    char *tp_8 = "餐饮服务;快餐厅;茶餐厅";
    char *tp_9 = "餐饮服务;外国餐厅;韩国料理";
    char *tp_10 = "购物服务;专卖店;摄影器材店";
    char *tp_11 = "风景名胜;公园广场;植物园";
    char *tp_12 = "汽车维修;现代特约维修;现代维修";
    char *tp_13 = "购物服务;文化用品店;文化用品店";
    char *tp_14 = "体育休闲服务;娱乐场所;夜总会";
    char *tp_15 = "科教文化服务;图书馆;图书馆";
    char *tp_16 = "餐饮服务;茶艺馆;茶艺馆";
    char *tp_17 = "购物服务;专卖店;眼镜店";
    char *tp_18 = "公司企业;公司;机械电子";
    char *tp_19 = "科教文化服务;档案馆;档案馆";
    char *tp_20 = "购物服务;专卖店;音像店";

    S2_POI p1;
    p1.set_keywords("");
    p1.set_type(tp_1);

    S2_POI p2;
    p2.set_keywords("");
    p2.set_type(tp_2);

    S2_POI p3;
    p3.set_keywords("");
    p3.set_type(tp_3);

    S2_POI p4;
    p4.set_keywords("");
    p4.set_type(tp_4);

    S2_POI p5;
    p5.set_keywords("");
    p5.set_type(tp_5);

    S2_POI p6;
    p6.set_keywords("");
    p6.set_type(tp_6);
        
    S2_POI p7;
    p7.set_keywords("");
    p7.set_type(tp_7);

    S2_POI p8;
    p8.set_keywords("");
    p8.set_type(tp_8);
        
    S2_POI p9;
    p9.set_keywords("");
    p9.set_type(tp_9);
        
    S2_POI p10;
    p10.set_keywords("");
    p10.set_type(tp_10);

    S2_POI p11;
    p11.set_keywords("");
    p11.set_type(tp_11);

    S2_POI p12;
    p12.set_keywords("");
    p12.set_type(tp_12);

    S2_POI p13;
    p13.set_keywords("");
    p13.set_type(tp_13);

    S2_POI p14;
    p14.set_keywords("");
    p14.set_type(tp_14);

    S2_POI p15;
    p15.set_keywords("");
    p15.set_type(tp_15);

    S2_POI p16;
    p16.set_keywords("");
    p16.set_type(tp_16);
        
    S2_POI p17;
    p17.set_keywords("");
    p17.set_type(tp_17);

    S2_POI p18;
    p18.set_keywords("");
    p18.set_type(tp_18);
        
    S2_POI p19;
    p19.set_keywords("");
    p19.set_type(tp_19);
        
    S2_POI p20;
    p20.set_keywords("");
    p20.set_type(tp_20);
    

    std::vector<S2_POI> poi_vec;
    poi_vec.push_back(p1);
    poi_vec.push_back(p2);
    poi_vec.push_back(p3);
    poi_vec.push_back(p4);
    poi_vec.push_back(p5);
    poi_vec.push_back(p6);
    poi_vec.push_back(p7);
    poi_vec.push_back(p8);
    poi_vec.push_back(p9);
    poi_vec.push_back(p10);
    poi_vec.push_back(p11);
    poi_vec.push_back(p12);
    poi_vec.push_back(p13);
    poi_vec.push_back(p14);
    poi_vec.push_back(p15);
    poi_vec.push_back(p16);
    poi_vec.push_back(p17);
    poi_vec.push_back(p18);
    poi_vec.push_back(p19);
    poi_vec.push_back(p20);

    return poi_vec;
}

void test_read_cell_points()
{
    //char *file_name = "/home/wangfei/201401TRK/TRK20140101/C02/C02668_n.txt";
    char *file_name = "/home/wangfei/06_nn/1000_n.txt";

    Movix mx;
    mx.read_cell_points(file_name);    
}

void test_build_index()
{
    Movix mx;
    RTree tree = mx.build_index_rtree("/home/wangfei/06_nn/");

    printf("depth: %lu, bbox: %llu, %llu, %lu, %lu\n", tree.depth, tree.root->bbox.left, tree.root->bbox.right, tree.root->bbox.top, tree.root->bbox.bottom);
}

void test_search()
{
    Movix mx;
    RTree tree = mx.build_index_rtree("/home/wangfei/06_n");

    std::vector<uint64> cid_vec;
    cid_vec.push_back(3769337298034890000l);
    cid_vec.push_back(3769340000000000000l);
    std::vector<int> points = 
            mx.search_RTree(&tree, cid_vec, 1388541772l, 1388600500l);

    for (int i = 0; i < points.size(); i++)
        printf("points %d: %d\n", i, points[i]);

}

void test_poi_search()
{
    POI_Data pd;
    pd.read_poi("beijing.csv");

    pd.batch_transform(11);
    
    char *tp = "科教文化服务;传媒机构;出版社";

    S2_POI p;
    p.set_keywords("");
    p.set_type(tp);

    std::vector<S2_POI> vec;
    vec.push_back(p);

    //std::vector<uint64> vec = pd.search(kw, tp);
    
    Movix mx;
    // HMix hm = mx.build_bitmap_index("/home/wangfei/11_06_n");
    HMix hm_12 = mx.build_bitmap_index("/home/wangfei/12_06_n");

    //size_t count = mx.poi_search(&hm, &pd, vec, 1202074000l, 1202285000, 25);
    //RTree tree = mx.build_index_rtree("/home/wangfei/06_n");

    std::vector<unsigned long> time_vec;
    time_vec.push_back(1202075000l);
    //1202076000l
    //1202077000l
    //1202078000l
    // size_t count = mx.poi_search(&hm, &hm_12, &pd, vec, time_vec, 25);

    // printf("-----%lu\n", count);
    
}

void test_multiple_poi_search()
{
    POI_Data pd;
    //pd.read_poi("test.csv");
    pd.read_poi("beijing.csv");

    pd.batch_transform(13);


    // char *tp_1 = "road";
    // char *tp_2 = "school;university";
    // char *tp_3 = "food";
    // char *tp_4 = "hospital";
    // char *tp_5 = "finacial"; 

    std::vector<S2_POI> poi_vec = set_poi();

    //long s = 1202074000;
    //long e = 1202285000;
   // std::vector<int> points = mx.poi_search(&tree, &pd, poi_vec, s, e);
    
    // printf("totally found %lu trajectories\n", points.size());

    //for (int i = 0; i < points.size(); ++i)
    //    printf("tid: %d\n", points[i]);

    Movix mx;
    HMix hm = mx.build_bitmap_index("/home/wangfei/13_06_n");
    // Movix mx_12;
    // HMix hm_12 = mx_12.build_bitmap_index("/home/wangfei/12_06_n");
    std::clock_t start;
    double duration;

    start = std::clock();

    std::vector<unsigned long> time_vec;
    time_vec.push_back(1202220000l);
    time_vec.push_back(1202221800l);    //0.5
    // time_vec.push_back(1202227200l);    //2hours

    // time_vec.push_back(1202248800l);//8hour
    // time_vec.push_back(1202306400l);//24hour
    // time_vec.push_back(1202325696l);
    time_vec.push_back(1202326002l);
    time_vec.push_back(1202491993l);
    // time_vec.push_back(1202342002l);
    //1202077000l
    //1202078000l
    size_t count = mx.poi_search(&hm, NULL, &pd, poi_vec, time_vec, 60);

    duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;

    std::cout<<"poi search cost: "<< duration <<'\n';

     printf("-----count: %lu\n", count);

}

void test_build_bitmap_index()
{
    Movix mx;
    HMix hm = mx.build_bitmap_index("/home/wangfei/11_06_nn");
    map<int, H_Index> v = hm.get_hmap();

    auto iter = v.begin(), end = v.end();
    for (; iter != end; ++iter)
    {
        iter->second.cell_bitmap->printf();
    }
}

void test_cost_by_time()
{
    // 20types, 11th level, 129.88s
    std::clock_t start;
    double duration;

    POI_Data pd;
    //pd.read_poi("test.csv");
    pd.read_poi("beijing.csv");

    pd.batch_transform(11);


    // char *tp_1 = "road";
    // char *tp_2 = "school;university";
    // char *tp_3 = "food";
    // char *tp_4 = "hospital";
    // char *tp_5 = "finacial"; 

    std::vector<S2_POI> poi_vec = set_poi();
 
    start = std::clock();

    Movix mx;
    HMix hm = mx.build_bitmap_index("/home/wangfei/11_06_n");

    duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;

    std::cout<<"build bitmap index cost: "<< duration <<'\n';

    // std::vector<std::vector<uint64> > V;
    // std::vector<uint64> cells_1, cells_2, cells_3, cells_4, cells_5;
    // cells_1.push_back(3886682874077773824);
    // cells_1.push_back(3886691013040799744);
    // cells_1.push_back(3887084170052108288);
    // cells_2.push_back(3886691966523539456);
    // cells_2.push_back(3886690628641226752);
    // cells_2.push_back(3886690630788710400);
    // cells_3.push_back(3886682204062875648);
    // cells_3.push_back(3886682485383233536);
    // cells_4.push_back(3887079239429652480);
    // cells_4.push_back(3887084172199591936);
    // cells_4.push_back(3887084185084493824);
    // cells_4.push_back(3887084208706813952);
    // cells_5.push_back(3887078782015635456);
    // cells_5.push_back(3887078743360929792);
    // cells_5.push_back(3887078734770995200);

    // V.push_back(cells_1);
    // V.push_back(cells_2);
    // V.push_back(cells_3);
    // V.push_back(cells_4);
    // V.push_back(cells_5);

    std::vector<std::vector<uint64> > cells_vec;
    
    // search in poi, return cell id
    for (int i = 0; i < poi_vec.size(); ++i)
    {
        std::vector<uint64> cid_vec = pd.search(poi_vec[i].keywords, poi_vec[i].type);

        printf("---poi cell: %lu\n", cid_vec.size());

        if (!cid_vec.empty())
            cells_vec.push_back(cid_vec);
    }

    start = std::clock();

    unsigned long *T = new unsigned long[2] {1201960197, 1202491993};
    std::map<int, H_Index> map = mx.cost_by_time(&hm, cells_vec, T, 25);

    duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;

    std::cout<<"estimate cost of time intervals cost: "<< duration <<'\n';

    printf("estimated count of time intervals: %lu", map.size());
}

void test_cost_by_types()
{
    std::clock_t start;
    double duration;

    POI_Data pd;
    // pd.read_poi("test.csv");
    pd.read_poi("beijing.csv");

    pd.batch_transform(11);


    // char *tp_1 = "road";
    // char *tp_2 = "school;university";
    // char *tp_3 = "food";
    // char *tp_4 = "hospital";
    // char *tp_5 = "finacial"; 

    std::vector<S2_POI> poi_vec = set_poi();

    start = std::clock();

    Movix mx;
    HMix hm = mx.build_bitmap_index("/home/wangfei/11_06_n");

    duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;

    std::cout<<"build bitmap index cost: "<< duration <<'\n';

    std::vector<std::vector<uint64> > cells_vec;
        std::vector<std::vector<uint64> > V;
    
    // search in poi, return cell id
    for (int i = 0; i < poi_vec.size(); ++i)
    {
        std::vector<uint64> cid_vec = pd.search(poi_vec[i].keywords, poi_vec[i].type);

        printf("---poi cell: %lu\n", cid_vec.size());

        if (!cid_vec.empty())
            cells_vec.push_back(cid_vec);
    }

    // std::vector<uint64> cells_1, cells_2, cells_3, cells_4, cells_5, cells_6, cells_7, cells_8, cells_9, cells_10, cells_11, cells_12, cells_13, cells_14, cells_15;
    // cells_1.push_back(3886682874077773824);
    // cells_1.push_back(3886691013040799744);
    // cells_1.push_back(3887084170052108288);
    // cells_2.push_back(3886691966523539456);
    // cells_2.push_back(3886690628641226752);
    // cells_2.push_back(3886690630788710400);
    // cells_3.push_back(3886682204062875648);
    // cells_3.push_back(3886682485383233536);
    // cells_4.push_back(3887079239429652480);
    // cells_4.push_back(3887084172199591936);
    // cells_4.push_back(3887084185084493824);
    // cells_4.push_back(3887084208706813952);
    // cells_5.push_back(3887078782015635456);
    // cells_5.push_back(3887078743360929792);
    // cells_5.push_back(3887078734770995200);
    // cells_6.push_back(3887079239429652480);
    // cells_6.push_back(3887084172199591936);
    // cells_6.push_back(3887084185084493824);
    // cells_7.push_back(3887084208706813952);
    // cells_7.push_back(3887078782015635456);
    // cells_7.push_back(3887078743360929792);
    // cells_7.push_back(3887078734770995200);
    // cells_8.push_back(3887079239429652480);
    // cells_8.push_back(3887084172199591936);
    // cells_8.push_back(3887084185084493824);
    // cells_9.push_back(3887084208706813952);
    // cells_9.push_back(3887078782015635456);
    // cells_9.push_back(3887078743360929792);
    // cells_9.push_back(3887078734770995200);
    // cells_10.push_back(3887079239429652480);
    // cells_10.push_back(3887084172199591936);
    // cells_10.push_back(3887084185084493824);
    // cells_11.push_back(3887084208706813952);
    // cells_11.push_back(3887079239429652480);
    // cells_11.push_back(3887084172199591936);
    // cells_12.push_back(3887084185084493824);
    // cells_12.push_back(3887084208706813952);
    // cells_12.push_back(3887078782015635456);
    // cells_13.push_back(3887078743360929792);
    // cells_13.push_back(3887078734770995200);
    // cells_13.push_back(3887079239429652480);
    // cells_14.push_back(3887084172199591936);
    // cells_14.push_back(3887084185084493824);
    // cells_14.push_back(3887084208706813952);
    // cells_15.push_back(3887078782015635456);
    // cells_15.push_back(3887078743360929792);
    // cells_15.push_back(3887078734770995200);


    // V.push_back(cells_1);
    // V.push_back(cells_2);
    // V.push_back(cells_3);
    // V.push_back(cells_4);
    // V.push_back(cells_5);
    // V.push_back(cells_6);
    // V.push_back(cells_7);
    // V.push_back(cells_8);
    // V.push_back(cells_9);
    // V.push_back(cells_10);
    // V.push_back(cells_11);
    // V.push_back(cells_12);
    // V.push_back(cells_13);
    // V.push_back(cells_14);
    // V.push_back(cells_15);

    start = std::clock();

    std::vector<float> v;
    std::vector<unsigned long> time_vec(5);
    // std::vector<std::pair<int, unsigned long> > count_vec = mx.cost_by_types(&hm, cells_vec, time_vec, v);

    duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;

    std::cout<<"estimate cost of types : "<< duration << "seconds \n";
    
    // for (int i = 0; i < count_vec.size(); ++i)
        // std::cout << "count number: " << count_vec.size() << std::endl;
}



int main()
{
    //test_read_cell_points();
    //test_build_index();
    //test_search();
    //test_poi_search();
    test_multiple_poi_search();
    // test_build_bitmap_index();
    // test_cost_by_time();
    // test_cost_by_types();
}
