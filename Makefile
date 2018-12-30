
CC = mpiicc
CXX = mpiicpc
CUDAC = nvcc

COPT =
CUDAOPT =

CFLAGS = -Wall -std=c99 $(COPT)
CXXFLAGS = -Wall -std=c++11 $(COPT)
CUDAFLAGS = $(CUDAOPT)

CUDAPATH = /usr/local/cuda/lib64
LDFLAGS = -Wall
LDLIBS = $(LDFLAGS)
GPULIBS = -L$(CUDAPATH) -L$(CUDAPATH)/stubs -lcuda -lcudart

targets = genstat benchmark-sequential benchmark-load-balance benchmark-gpu-nvgraph
commonobj = utils.o benchmark.o
objects = $(commonobj) graph-sequential.o graph-load-balance.o graph-gpu-nvgraph.o graph-mysssp.o

.PHONY : default
default : all

.PHONY : all
all : clean $(targets) stat.csv

benchmark.o : benchmark.c common.h utils.h
	$(CC) -c $(CFLAGS) $< -o $@
utils.o : utils.c common.h
	$(CC) -c $(CFLAGS) -fp-model=strict $< -o $@

graph-sequential.o : graph-sequential.c common.h
	$(CC) -c $(CFLAGS) $< -o $@
benchmark-sequential : $(commonobj) graph-sequential.o
	$(CC) -o $@ $^ $(LDLIBS)

graph-mysssp.o : graph-mysssp.c common.h
	$(CC) -c $(CFLAGS) $< -o $@
benchmark-mysssp : $(commonobj) graph-mysssp.o
	$(CC) -o $@ $^ $(LDLIBS)

graph-load-balance.o : graph-load-balance.c common.h
	$(CC) -c $(CFLAGS) $< -o $@
benchmark-load-balance : $(commonobj) graph-load-balance.o
	$(CC) -o $@ $^ $(LDLIBS)

graph-gpu-nvgraph.o : graph-gpu-nvgraph.cu common.h utils.h
	$(CUDAC) -c $(CUDAFLAGS) $< -o $@
benchmark-gpu-nvgraph : $(commonobj) graph-gpu-nvgraph.o
	$(CC) -o $@ $^ $(LDLIBS) $(GPULIBS) -qopenmp -lnvgraph

.PHONY: clean
clean:
	rm -rf $(targets) $(objects)
	
genstat:genstat.c
	icc $(CFLAGS) $< -o $@

stat.csv:genstat genstat.sh
	bash genstat.sh