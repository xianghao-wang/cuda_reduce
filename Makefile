NVCC := nvcc

EXES := reduce_opt0 \
	reduce_opt1 \
	reduce_opt2 \
	reduce_opt3

all: $(EXES)

reduce_opt0: reduce.cu
	$(NVCC) -D__OPT__=0 -o $@ $< 

reduce_opt1: reduce.cu
	$(NVCC) -D__OPT__=1 -o $@ $< 

reduce_opt2: reduce.cu
	$(NVCC) -D__OPT__=2 -o $@ $< 

reduce_opt3: reduce.cu
	$(NVCC) -D__OPT__=3 -o $@ $< 

clean: FORCE 
	rm -f $(EXES)

FORCE: ;