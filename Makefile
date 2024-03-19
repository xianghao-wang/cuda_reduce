NVCC := nvcc

TB_SIZE ?= 1024
NITERS ?= 1024

NVCC_FLAGS += -DTB_SIZE=$(TB_SIZE) -DNITERS=$(NITERS)

ANNOTATE_FLAGS := -G --ptx --source-in-ptx

EXES := reduce_opt0 reduce_opt1 reduce_opt2 reduce_opt3

all: $(EXES) reduce.ptx

reduce_opt0: reduce.cu
	$(NVCC) $(NVCC_FLAGS) -D__OPT__=0 -o $@ $<

reduce_opt1: reduce.cu
	$(NVCC) $(NVCC_FLAGS) -D__OPT__=1 -o $@ $<

reduce_opt2: reduce.cu
	$(NVCC) $(NVCC_FLAGS) -D__OPT__=2 -o $@ $<

reduce_opt3: reduce.cu
	$(NVCC) $(NVCC_FLAGS) -D__OPT__=3 -o $@ $<

reduce.ptx: reduce.cu
	$(NVCC) $(ANNOTATE_FLAGS) -D__OPT__=0 -o $@ $<

clean: FORCE 
	rm -rf $(EXES) ptxs *.ptx *.out

FORCE: ;