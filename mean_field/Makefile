include Makefile.conf

# Define build targets
MATLAB_TARGETS: build/calcF$(ME) build/calc_alpha$(ME)
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

build/calc_alpha$(ME) : mex/calc_alpha.cpp
	 $(MEX) $(MEXOUTFLAG) $@ mex/calc_alpha.cpp

# CLEAN
########
.PHONY : clean
clean :
	-rm build/*$(ME)
