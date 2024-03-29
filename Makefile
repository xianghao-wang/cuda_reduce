NVCC := nvcc

TB_SIZE ?= 1024
NITERS ?= 8

NVCC_FLAGS += -O3 -DTB_SIZE=$(TB_SIZE) -DNITERS=$(NITERS)

ANNOTATE_FLAGS := -lineinfo --ptx --source-in-ptx

EXES := reduce_opt0 reduce_opt1 reduce_opt2 reduce_opt3 reduce_opt4

all: $(EXES)

reduce_opt0: reduce.cu
	$(NVCC) $(NVCC_FLAGS) -D__OPT__=0 -o $@ $<

reduce_opt1: reduce.cu
	$(NVCC) $(NVCC_FLAGS) -D__OPT__=1 -o $@ $<

reduce_opt2: reduce.cu
	$(NVCC) $(NVCC_FLAGS) -D__OPT__=2 -o $@ $<

reduce_opt3: reduce.cu
	$(NVCC) $(NVCC_FLAGS) -D__OPT__=3 -o $@ $<

reduce_opt4: reduce.cu
	$(NVCC) $(NVCC_FLAGS) -D__OPT__=4 -o $@ $<

ptx: reduce.cu
	$(NVCC) $(ANNOTATE_FLAGS) -D__OPT__=0 -o $@ $<

SIZE ?= 16777216
prof: $(EXES)
	mkdir -p profiles
	for exe in $(EXES) ; do \
		ncu --set full -f -o profiles/profile_$$exe ./$$exe $(SIZE) ; \
	done 
	ncu-ui profiles/*

clean: FORCE 
	rm -rf $(EXES) ptxs *.ptx *.out profiles

FORCE: ;