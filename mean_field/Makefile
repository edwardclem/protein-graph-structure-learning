include Makefile.conf

# Define build targets
MATLAB_TARGETS: build/calcF$(ME) build/calc_muhat$(ME) build/calc_muhat_parallel$(ME) build/calc_muhat_parallel_cond$(ME) build/calcPll$(ME)
TARGETS:=$(TARGETS) $(MATLAB_TARGETS)

MEX:=$(MEX) $(MEXFLAGS) $(MEXINC) $(MEXLIB) $(MEXLIBS)

# META TARGETS
###############

all : $(TARGETS)
	@echo
	@echo built successfully!

matlabs : $(MATLAB_TARGETS)

# MATLAB CODE
###################
build/calcF$(ME) : mex/calcF.cpp
	 $(MEX) $(MEXOUTFLAG) $@ mex/calcF.cpp

build/calc_muhat$(ME) : mex/calc_muhat.cpp
	 $(MEX) $(MEXOUTFLAG) $@ mex/calc_muhat.cpp

build/calc_muhat_parallel$(ME) : mex/calc_muhat_parallel.cpp
	 $(MEX) $(MEXOUTFLAG) $@ mex/calc_muhat_parallel.cpp

build/calc_muhat_parallel_cond$(ME) : mex/calc_muhat_parallel_cond.cpp
	 $(MEX) $(MEXOUTFLAG) $@ mex/calc_muhat_parallel_cond.cpp

build/calcPll$(ME) : mex/calcPll.cpp
	$(MEX) $(MEXOUTFLAG) $@ mex/calcPll.cpp
	
# CLEAN
########
.PHONY : clean
clean :
	-rm build/*$(ME)
