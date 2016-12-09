fprintf('Compiling mean_field CRF files...\n');
mex -outdir ./build ./mex/calcF.cpp CXXFLAGS="\$CXXFLAGS -std=c++11"
mex -outdir ./build ./mex/calc_muhat.cpp CXXFLAGS="\$CXXFLAGS -std=c++11"
mex -outdir ./build ./mex/calc_muhat_parallel.cpp CXXFLAGS="\$CXXFLAGS -std=c++11"