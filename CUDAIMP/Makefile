#vim: set noet
PROG=2048
WB_PATH=/work/ece497/sp1920/libwb
OPTS=-std=c++11 -arch=sm_30 -I $(WB_PATH) $(WB_PATH)/libwb.a
CC=nvcc

all: $(PROG) dataset_generator


$(PROG): $(PROG).cu
	$(CC) $(OPTS) $^ -o $@ 

debug: $(PROG).cu
	$(CC) -g -G -Xcompiler -rdynamic $(OPTS) $^ -o $(PROG)_$@ 



