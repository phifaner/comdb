#ifndef _POI_H
#define _POI_H

#include <vector>
#include <map>
#include <string.h>

#include "s2/base/integral_types.h"

// a POI record
struct POI
{
    char *name;
    char *type;
    float longitude, latitude;
};

// transform one POI into S2 style
class S2_POI
{
  public:
    void transform(POI *poi, int l);

    inline void set_keywords(char *kw) { keywords = kw; }

    inline void set_type(char *tp) { type = tp; }

    inline void set_cell_id(uint64 cid) { cell_id = cid; }

    inline uint64 get_cell_id() { return cell_id; }

  //private:
    char *keywords;
    char *type;
    uint64 cell_id;

    bool operator==(S2_POI p)
    {
        if ( (strcmp(keywords, p.keywords) == 0) &&
             (strcmp(type, p.type) == 0) &&
             (cell_id == p.cell_id) )
            return true;

        return false;
    }
};

struct cmp_str
{
  
   bool operator() (char const *a,  char const *b)
   {
        if (strcmp(a, b) < 0 )
          return true;
      // return strcmp(a, b) == 0;
   }
};


// operate all POI of S2 style
class POI_Data
{
  public:
    std::map<const std::string, std::vector<POI> > poi_map;
    // std::map<const char*, std::vector<uint64>, cmp_str> s2_poi_map;
    std::map<const std::string, std::vector<uint64> > s2_poi_map;

    void read_poi(char *filename);

    void batch_transform(int l);

    std::vector<uint64> search_s2_poi_by_type(char *tp);

    std::vector<POI> search_poi_by_type(char *tp);

    std::map<const std::string, std::vector<POI> > get_all_poi_types();

    POI search_poi_by_type_keyword(char *type, char *keyword);

};
#endif
