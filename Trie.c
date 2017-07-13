#include "Trie.h"
#include "s2/s2cellid.h"
#include "s2/s1angle.h"
#include "s2/s2latlng.h"


#include <math.h>
#include <stdio.h>

Trie::Trie(double v, unsigned long t)
: V(v), T(t)
{
    root = new Ti_Node();
}

Trie::~Trie()
{
    // Free memory
    root = NULL;
    delete root;
}

double measure(double lat1, double lon1, double lat2, double lon2)
{
    double R = 6378.137;     // radius of earth in KM
    double d_lat = lat2 * M_PI / 180 - lat1 * M_PI / 180;
    double d_lon = lon2 * M_PI / 180 - lon1 * M_PI / 180;
    double a = sin(d_lat / 2) * sin(d_lat / 2) +
            cos(lat1 * M_PI / 180) * cos(lat2 * M_PI / 180) *
            sin(d_lon / 2) * sin(d_lon / 2);
    double c = 2 * atan2(sqrt(a), sqrt(1-a));
    double d = R * c;
    return d * 1000;
}

// add children to node
void Trie::insert(struct Ti_Node * node, vector<uint64_t> cells, char * type, double V, double T)
{
	struct Ti_Node * n = node;

	vector<uint64_t>::iterator it;
	for (it = cells.begin(); it != cells.end(); ++it)
	{

		cout << "node id: " << n->cell_id << endl;
		cout << "cell id: " << *it << endl;

		// no children
		if (n->children.size() == 0)
		{
			// compute distance between current node and each child
			S2CellId s1(*it);
            S2CellId s2(n->cell_id);
            S2LatLng ll1 = s1.ToLatLng();
            S2LatLng ll2 = s2.ToLatLng();
            S1Angle lat1 = ll1.lat();
            S1Angle lon1 = ll1.lng();
            S1Angle lat2 = ll2.lat();
            S1Angle lon2 = ll2.lng();
            double d = measure(lat1.degrees(), lon1.degrees(), lat2.degrees(), lon2.degrees());

            // if total distance less than v * T, add in children
			if (n->distance +  d < V * T || n->cell_id == 0)
			{
				Ti_Node *cn = (struct Ti_Node *) calloc(1, sizeof(struct Ti_Node));
				cn->cell_id = *it;

				cn->type = type;
				cn->parent = n;
				n->children.push_back(cn);
			}
		}
	}
}

vector<Edge> Trie::multix(vector<Edge> L, vector<uint64> R)
{
	vector<Edge> VE;

	vector<Edge>::iterator lit, lend;
	vector<uint64>::iterator rit, rend;
	for (lit = L.begin(), lend = L.end(); lit != lend; ++lit)
	{
		for (rit = R.begin(), rend = R.end(); rit != rend; ++rit)
		{
			// get the last cell id
			S2CellId s1(lit->cells.back());
            S2CellId s2(*rit);
            S2LatLng ll1 = s1.ToLatLng();
            S2LatLng ll2 = s2.ToLatLng();
            S1Angle lat1 = ll1.lat();
            S1Angle lon1 = ll1.lng();
            S1Angle lat2 = ll2.lat();
            S1Angle lon2 = ll2.lng();
            double d = measure(lat1.degrees(), lon1.degrees(), lat2.degrees(), lon2.degrees());

            //cout << lat1.degrees() << ", " << lon1.degrees() << " | " << lat2.degrees() << ", " << lon2.degrees() << endl;

			// cout << "distance: " << V * T << "|" << lit->distance + d << endl;

            // if satisfied, add the cell into Edge
			if (lit->distance + d < V * T) 
			{
				Edge e;
				e.cells.insert(e.cells.begin(), lit->cells.begin(), lit->cells.end());
				e.cells.push_back((uint64_t)*rit);
				e.distance = lit->distance + d;

				VE.push_back(e);
			}
		}
	}

	L.clear();

	return VE;
}


vector<Measure> Trie::multix(vector<Measure> L, vector<Pair> R)
{
	vector<Measure> VM;

	vector<Measure>::iterator lit, lend;
	vector<Pair>::iterator rit, rend;
	for (lit = L.begin(), lend = L.end(); lit != lend; ++lit)
	{
		for (rit = R.begin(), rend = R.end(); rit != rend; ++rit)
		{
			// get the latest timestamp
			unsigned long latest = lit->latest;
			unsigned long ts = rit->time;

            //cout << lat1.degrees() << ", " << lon1.degrees() << " | " << lat2.degrees() << ", " << lon2.degrees() << endl;

            // if satisfied, add the cell into Edge
			if (ts <= latest) 
			{
				lit->cells.push_back(rit->cell_id);
			} 
			else  
			{
				unsigned long d = ts - latest;
				if (lit->interval + d < T)
				{
					// if less than time constraint, change copy L's cells and add new cell
					Measure m;
					m.cells.insert(m.cells.begin(), lit->cells.begin(), lit->cells.end());
					m.cells.push_back(rit->cell_id);
					m.interval = lit->interval + d;
					m.latest = ts;

					VM.push_back(m);
				}
			}
		}
	}

	return VM;
}

void Trie::output(struct Ti_Node *tree)
{

}

void Trie::build(vector<char *> T, vector<vector<uint64_t> > C)
{
	assert (T.size() == C.size());

	vector<Ti_Node *> leaves, container;
	root->cell_id = 0;
	leaves.push_back(root);

	// BFS insert nodes
	size_t idx, end;
	for (idx = 0, end = C.size(); idx != end; ++idx)
	{
		while (!leaves.empty())
		{
			Ti_Node *node = leaves.back();
			leaves.pop_back();
			insert(node, C[idx], T[idx], 25.0, 2*3600);

			// add new children as leaves
			container.insert(container.end(), node->children.begin(), 
				node->children.end());
		}
	
		leaves = container;	
	}
}