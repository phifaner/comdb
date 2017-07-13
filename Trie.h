#include <stdint.h>
#include <vector>

#include "s2/base/integral_types.h"

using namespace std;

struct Ti_Node
{
	struct Ti_Node * parent;
	vector<Ti_Node *> children;
	vector<int> occurr;
	uint64_t cell_id;			// cell id of S2
	double distance;			// distance from root to this node
	char * type;				// POI type
};

/**
 * we use edge as a measure of all types of cells
**/
struct Edge
{
	vector<uint64> cells;
	
	double distance;
};

struct Measure
{
	vector<uint64> cells;

	// the latest time
	unsigned long latest;

	unsigned long interval;
};

struct Pair
{
	uint64 cell_id;
	unsigned long time;
};

class Trie
{
  public:
  	Trie(double v, unsigned long t);
  	~Trie();
	
	// if distance is not 
	void insert(struct Ti_Node * node, vector<uint64_t> cells, char * type, double V, double T);
	void output(struct Ti_Node *tree);

	void build(vector<char *> T, vector<vector<uint64_t> > C);

	// compute cell's distance when it < V*T
	vector<Edge> multix(vector<Edge> L, vector<uint64> R);

	// compute cell's last time when it < T
	vector<Measure> multix(vector<Measure> L, vector<Pair> R);
	
  private:
	Ti_Node * root;

	double V;
	unsigned long T;
};

