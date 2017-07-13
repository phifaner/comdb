#include "s2/s2cellid.h"
#include "s2/s2latlng.h"
#include "s2/s1angle.h"

#include <stdio.h>

int main()
{
	uint64 cid = 3886896405893087232; //12th; 3886698837397471232;//9th;3887075145252077568; //11th; //3886694439350960128;//3886698837397471232;//3887099059629981696;//3886676847164915712;
	S2CellId cell(cid);

	// int i = 0;
	// for (S2CellId c = cell.child_begin(12); c != cell.child_end(12); c = c.next())
	// {
	// 	printf("----%llu\n", c.id());

	// 	i++;
	// }

	// printf("----%d\n", i);

	S2LatLng ll = cell.ToLatLng();
	S2CellId n = cell.next();
	S2LatLng ll_n = n.ToLatLng();

	S1Angle lat = ll.lat()-ll_n.lat();
	S1Angle lon = ll.lng()-ll_n.lng();

	printf("lat: %lf, lon: %lf\n", lat.abs().radians(), lon.abs().radians());
	printf("ll lat: %lf, ll lon: %lf, ll_n lat: %lf, ll_n lon: %lf\n", ll.lat().degrees(), ll.lng().degrees(), ll_n.lat().degrees(), ll_n.lng().degrees());
}