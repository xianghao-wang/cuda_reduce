NVCC := nvcc

TB_SIZE ?= 1024
NITERS ?= 8

NVCC_FLAGS += -DTB_SIZE=$(TB_SIZE) -DNITERS=$(NITERS)

ANNOTATE_FLAGS := -lineinfo --ptx --source-in-ptx

EXES := reduce_opt0 reduce_opt1 reduce_opt2

all: $(EXES)

reduce_opt0: reduce.cu
	$(NVCC) $(NVCC_FLAGS) -D__OPT__=0 -o $@ $<

reduce_opt1: reduce.cu
	$(NVCC) $(NVCC_FLAGS) -D__OPT__=1 -o $@ $<

reduce_opt2: reduce.cu
	$(NVCC) $(NVCC_FLAGS) -D__OPT__=2 -o $@ $<

ptx: reduce.cu
	$(NVCC) $(ANNOTATE_FLAGS) -D__OPT__=0 -o $@ $<

SIZE ?= 16777216
prof: $(EXES)
	mkdir -p profiles
	for exe in $(EXES) ; do \
		ncu --set full -f -o profiles/profile_$$exe ./$$exe $(SIZE) ; \
	done 

clean: FORCE 
	rm -rf $(EXES) ptxs *.ptx *.out profiles

FORCE: ;