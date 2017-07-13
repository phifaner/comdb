#include "Trie.h"
#include <iostream>

void test_build()
{
	Trie t(25.0, 3600);

	vector<char *> T;
	T.push_back("school");
	T.push_back("hotel");
	T.push_back("company");

	vector<vector<uint64_t> > C;
	vector<uint64_t> t1, t2, t3;
	t1.push_back(3886994128210231296);
	t1.push_back(3886994169012420608);
	t1.push_back(3886993917756833792);
	t1.push_back(3886698082556968960);

	t2.push_back(3886696869228707840);
	t2.push_back(3886696957275537408);
	t2.push_back(3886698499168796672);
	t2.push_back(3886696950833086464);

	t3.push_back(3886696186328907776);
	t3.push_back(3886696175591489536);
	t3.push_back(3886700940857704448);
	t3.push_back(3886696452616880128);
	t3.push_back(3886692548491608064);

	C.push_back(t1);
	C.push_back(t2);
	C.push_back(t3);

	t.build(T, C);

}

void test_multiply()
{
	Trie t(25.0, 3600);

	vector<Edge> E;
	vector<uint64_t> t1, t2, t3;
	Edge e1, e2, e3, e4;
	e1.cells.push_back(3886994128210231296);
	e1.distance = 0;
	
	e2.cells.push_back(3886994169012420608);
	e2.distance = 0;
	e3.cells.push_back(3886993917756833792);
	e3.distance = 0;
	e4.cells.push_back(3886698082556968960);
	e4.distance = 0;

	E.push_back(e1);
	E.push_back(e2);
	E.push_back(e3);
	E.push_back(e4);

	t2.push_back(3886696869228707840);
	t2.push_back(3886696957275537408);
	t2.push_back(3886698499168796672);
	t2.push_back(3886696950833086464);

	t3.push_back(3886696186328907776);
	t3.push_back(3886696175591489536);
	t3.push_back(3886700940857704448);
	t3.push_back(3886696452616880128);
	t3.push_back(3886692548491608064);

	E = t.multix(E, t2);
	E = t.multix(E, t3);

	vector<Edge>::iterator ite;
	vector<uint64_t>::iterator itc;
	int count = 0;
	for (ite = E.begin(); ite != E.end(); ++ ite)
	{
		for (itc = ite->cells.begin(); itc != ite->cells.end(); ++itc)
		{
			cout << "cell id: " << *itc << endl;
		}

		count ++;
		cout << count << "----------\n" << endl;
	}
}

int main()
{
	//test_build();
	test_multiply();
}