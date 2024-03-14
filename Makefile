NVCC := nvcc

TB_SIZE ?= 1024
NITERS ?= 1024

NVCC_FLAGS += -DTB_SIZE=$(TB_SIZE) -DNITERS=$(NITERS)

EXES := reduce_opt0 reduce_opt1 reduce_opt2 reduce_opt3

all: $(EXES) ptxs $(addprefix ptxs/, $(addsuffix .ptx, $(EXES)))

reduce_opt0: reduce.cu
	$(NVCC) $(NVCC_FLAGS) -D__OPT__=0 -o $@ $<

reduce_opt1: reduce.cu
	$(NVCC) $(NVCC_FLAGS) -D__OPT__=1 -o $@ $<

reduce_opt2: reduce.cu
	$(NVCC) $(NVCC_FLAGS) -D__OPT__=2 -o $@ $<

reduce_opt3: reduce.cu
	$(NVCC) $(NVCC_FLAGS) -D__OPT__=3 -o $@ $<

ptxs: FORCE
	mkdir -p $@

ptxs/reduce_opt0.ptx: reduce.cu
	$(NVCC) --ptx $(NVCC_FLAGS) -D__OPT__=0 -o $@ $<

ptxs/reduce_opt1.ptx: reduce.cu
	$(NVCC) --ptx $(NVCC_FLAGS) -D__OPT__=1 -o $@ $<

ptxs/reduce_opt2.ptx: reduce.cu
	$(NVCC) --ptx $(NVCC_FLAGS) -D__OPT__=2 -o $@ $<

ptxs/reduce_opt3.ptx: reduce.cu
	$(NVCC) --ptx $(NVCC_FLAGS) -D__OPT__=3 -o $@ $<

clean: FORCE 
	rm -rf $(EXES) ptxs *.ptx

FORCE: ;