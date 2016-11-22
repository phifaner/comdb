# Linux Makefile

#uncomment for redis data dictionary [SIMPLE or HA] (default local)
#REDIS_FLAGS=-I ../deps/redis3m/include/ -DDATADICT_REDIS_SIMPLE=1
#REDIS_LIBS=../deps/redis3m/libredis3m.a -lhiredis -lboost_regex
#REDIS_OBJ=datadict_redis_ha.o datadict_redis_simple.o

#uncomment for hdfs filesystem (default local)
#HDFS_FLAGS=-DFILESYSTEM_HDFS=1 
#HDFS_LIBS=-lhdfs
#HDFS_OBJ=filesystem_hdfs.o

#cflags
CFLAGS=--machine 64 -g -G -std=c++11 --expt-extended-lambda --compiler-options '-fPIC' -I ../deps/ -I ../include/ $(REDIS_FLAGS) $(HDFS_FLAGS)

#compute compatibility (https://developer.nvidia.com/cuda-gpus)
#GENCODE_SM30	:= -gencode arch=compute_30,code=sm_30
#GENCODE_SM35	:= -gencode arch=compute_35,code=sm_35
GENCODE_SM52	:= -gencode arch=compute_52,code=sm_52
GENCODE_FLAGS	:= $(GENCODE_SM30) $(GENCODE_SM35) $(GENCODE_SM52)

#comdb binary
comdbs : main.o comdb.o  
	nvcc  -L . $^ $(REDIS_LIBS) $(HDFS_LIBS) -o $@ 

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
main.o : main.cu swindow.h
	$(nvcc) -c $< -o $@
comdb.o : comdb.cu comdb.h
	$(nvcc) -c $< -o $@
#heatmap.o : heatmap.cu heatmap.h
#	$(nvcc) -c $< -o $@
#swindow.o : swindow.cu swindow.h
#	$(nvcc) -c $< -o $@
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
