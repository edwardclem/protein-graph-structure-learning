function theta = minimizeLL_SGD(ss_proteins, L, N, features_aa, seqlen_all, crfOpt)

maxIter = 3; % Number of passes through the data set
stepSize = 1e-9;
theta = zeros([size(ss_proteins, 1), 1]);
for iter = 1:maxIter*L
    inds = randsample(L, crfOpt.nThreads); % Parallelized should work now
    [f,g] = getLlikCRFMean(theta, sum(ss_proteins(:,inds), 2), ...
        crfOpt.nThreads, N(inds), features_aa(inds), seqlen_all(inds), ...
        crfOpt);
    
    fprintf('\tIter = %d of %d (fsub = %f)\n',iter,maxIter*L,f);
    
    theta = theta - stepSize*g;
end

end