include Makefile.conf

# Define build targets
MATLAB_TARGETS: build/mean_field$(ME)
TARGETS:=$(TARGETS) $(MATLAB_TARGETS)
SRCS:=mean_field.cpp

DEPS:=$(SRCS:%.cpp=mex/%.cpp)
MEX:=$(MEX) $(MEXFLAGS) $(MEXINC) $(MEXLIB) $(MEXLIBS)

# META TARGETS
###############

all : $(TARGETS)
	@echo
	@echo built successfully!

matlabs : $(MATLAB_TARGETS)

# MATLAB CODE
###################
build/mean_field$(ME) : $(DEPS)
	 $(MEX) $(MEXOUTFLAG) $@ $(DEPS)

# CLEAN
########
.PHONY : clean
clean :
	-rm *$(ME)
