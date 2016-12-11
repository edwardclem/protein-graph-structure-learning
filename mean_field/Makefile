include Makefile.conf

# Define build targets
MATLAB_TARGETS: build/calcF$(ME) build/calc_muhat$(ME) build/calc_muhat_parallel$(ME)
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

# CLEAN
########
.PHONY : clean
clean :
	-rm build/*$(ME)
