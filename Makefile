# Linux Makefile

#uncomment for redis data dictionary [SIMPLE or HA] (default local)
#REDIS_FLAGS=-I ../deps/redis3m/include/ -DDATADICT_REDIS_SIMPLE=1
#REDIS_LIBS=../deps/redis3m/libredis3m.a -lhiredis -lboost_regex
#REDIS_OBJ=datadict_redis_ha.o datadict_redis_simple.o
S2_FLAGS=-I ./include

#uncomment for hdfs filesystem (default local)
#HDFS_FLAGS=-DFILESYSTEM_HDFS=1 
S2_LIBS=-ls2 -ls2cellid -ls2util
#HDFS_OBJ=filesystem_hdfs.o
WARNINGS=-Wno-deprecated-gpu-targets
CUDA_LIBS = -lcudadevrt
#cflags
#CFLAGS=--machine 64 -O2 -g -G -std=c++11 $(WARNINGS) -rdc=true --expt-extended-lambda --compiler-options '-fPIC' -I ./include/
CFLAGS=--machine 64 -g -std=c++11 $(WARNINGS) -rdc=true --expt-extended-lambda --compiler-options '-fPIC' -I ../deps/ $(S2_FLAGS)

#compute compatibility (https://developer.nvidia.com/cuda-gpus)
#GENCODE_SM30	:= -gencode arch=compute_30,code=sm_30
GENCODE_SM35	:= -gencode arch=compute_35,code=sm_35
GENCODE_SM52	:= -gencode arch=compute_52,code=sm_52
GENCODE_FLAGS	:= $(GENCODE_SM30) $(GENCODE_SM35) $(GENCODE_SM52)

#comdb binary
comdbs : test_s2_index.o comdb.o parser.o rtree.o poi.o construct.o mix.o s2_index.o trie.o file_index.o hash_table.o 
	nvcc  -L . $^ $(S2_LIBS) $(CUDA_LIBS) $(WARNINGS) -o $@ 

#static library
#libalenka.a : alenka.o callbacks.o common.o compress.o cudaset.o datadict_local.o \
		 $(REDIS_OBJ) $(HDFS_OBJ) filesystem_local.o filter.o global.o merge.o murmurhash2_64.o \
		 operators.o select.o sorts.o strings_join.o strings_sort_host.o strings_sort_device.o zone_map.o	 
#	ar rcs $@ $^

#shared library
#libalenka.so : alenka.o callbacks.o common.o compress.o cudaset.o datadict_local.o \
		 $(REDIS_OBJ) $(HDFS_OBJ) filesystem_local.o filter.o global.o merge.o murmurhash2_64.o \
		 operators.o select.o sorts.o strings_join.o strings_sort_host.o strings_sort_device.o zone_map.o	 
#	nvcc -o libalenka.so --shared $^

#alenka objects
nvcc = nvcc $(CFLAGS) $(GENCODE_FLAGS)
#main.o : main.cu  
#	$(nvcc) -c $< -o $@
test_s2_index.o : test_s2index.cu
	$(nvcc) -c $< -o $@
comdb.o : comdb.cu comdb.h
	$(nvcc) -c $< -o $@ 
parser.o : wzparser.cu 
	$(nvcc) -c $< -o $@ 
dbscan.o : dbscan.cu
	$(nvcc) -c $< -o $@
#quadtree.o : quadtree.cu
s2_index.o : s2_index.cu s2_index.h
	$(nvcc) -c $< -o $@
#heatmap.o : heatmap.cu
#	$(nvcc) -c $< -o $@
#swindow.o : swindow.cu swindow.h
#	$(nvcc) -c $< -o $@
project.o : project.cu Random.h
	$(nvcc) -c $< -o $@
random.o : Random.cpp Random.h
	gcc -c $< -o $@
file_index.o : file_index.c file_index.h
	g++ -c $< -o $@
hash_table.o : hash_table.c hash_table.h
	g++ -c $< -o $@
trie.o : Trie.c Trie.h
	g++  $(S2_FLAGS) -std=c++11 -c $< -o $@
mix.o : mix.cpp mix.h
	g++ $(S2_FLAGS) -std=c++11 -DDISABLE_X64=ON -c $< -o $@
construct.o : construct.c construct.h
	g++  $(S2_FLAGS) -std=c++11 -c $< -o $@
poi.o : poi.c poi.h
	g++  $(S2_FLAGS) -std=c++11 -c $< -o $@
rtree.o : rtree.cu rtree.h
	$(nvcc) -c $< -o $@
#bison / flex generation
#alenka.cu alenka.hu: alenka.ypp
#	bison -d -o $@ $<
#lex.yy.c: alenka.l alenka.hu
#	flex $<

#clean up all generated files
.PHONY: clean
clean : 
	rm -f comdb *.o 
	
#install binary + libs + header
#PREFIX = /usr/local
#.PHONY: install
#install: alenka libalenka.a libalenka.so
#	mkdir -p $(DESTDIR)$(PREFIX)/bin
#	cp $< $(DESTDIR)$(PREFIX)/bin/alenka
#	mkdir -p $(DESTDIR)$(PREFIX)/lib
#	mkdir -p $(DESTDIR)$(PREFIX)/include
#	cp libalenka.a $(DESTDIR)$(PREFIX)/lib64/
#	cp libalenka.so $(DESTDIR)$(PREFIX)/lib64/
#	cp ../include/alenka.h $(DESTDIR)$(PREFIX)/include/

#.PHONY: uninstall
#uninstall:
#	rm -f $(DESTDIR)$(PREFIX)/bin/alenka
#	rm -f $(DESTDIR)$(PREFIX)/lib64/libalenka.a
#	rm -f $(DESTDIR)$(PREFIX)/lib64/libalenka.so
#	rm -f $(DESTDIR)$(PREFIX)/include/alenka.h
