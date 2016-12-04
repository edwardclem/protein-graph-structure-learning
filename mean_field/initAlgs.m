function [algs] = initAlgs(N,nFeats)

    % model each class independently
    algs(1).name = 'independent';
    algs(1).totFeats = N;
    algs(1).inferFn = @baseInfer;

    % model each class according to an MRF and do exact gradient descent
    algs(2).name = 'mrf';
    algs(2).suffStatFn = @suffStatsMRF;
    algs(2).isingFn = @getLlikMRF;
    algs(2).totFeats = N*(N+1)/2;
    algs(2).inferFn = @mrfInfer;

    % model each class to a CRF and do exact gradient descent
    algs(3).name = 'crf';
    algs(3).suffStatFn = @suffStatsCRF;
    algs(3).isingFn = @getLlikCRF;
    algs(3).totFeats = N*(N+1)/2+N*nFeats;
    algs(3).inferFn = @crfInfer;
    
    % model each class using logistic regression
    algs(4).name = 'logistic';
    algs(4).suffStatFn = @suffStatsLogistic;
    algs(4).isingFn = @getLlikLogistic;
    algs(4).totFeats = N+N*nFeats;
    algs(4).inferFn = @logisticInfer;
    
    % model each class according to an MRF and do approximate gradient
    % descent
    algs(5).name = 'mrfMean';
    algs(5).suffStatFn = @suffStatsMRF;
    algs(5).isingFn = @getLlikMRFMean;
    algs(5).totFeats = N*(N+1)/2;
    algs(5).inferFn = @mrfMeanInfer;
    
    % model each class according to a CRF and do approximate gradient
    % descent
    algs(6).name = 'crfMean';
    algs(6).suffStatFn = @suffStatsCRF;
    algs(6).isingFn = @getLlikCRFMean;
    algs(6).totFeats = N*(N+1)/2+N*nFeats;
    algs(6).inferFn = @crfMeanInfer;
    
end

