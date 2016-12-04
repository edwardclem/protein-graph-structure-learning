function [algs,aucAll,llTrace] = runexp(dataset,algsInd)
    options.maxIter=1000;
    
    load(sprintf('q%d_data',dataset),'featTrain','featTest','objectsPresentTrain','objectsPresentTest','names');

    [N,L] = size(objectsPresentTrain);
    nFeats = size(featTrain,1);
    algs = initAlgs(N,nFeats);
    algs = algs(algsInd);
    
    lambdaBar = 0.01;
    llTrace = NaN(options.maxIter, numel(algs));

    for(a=1:numel(algs))

        theta = zeros(algs(a).totFeats,1);
        if(~strcmp(algs(a).name, 'independent'))
            
            [ss] = algs(a).suffStatFn(objectsPresentTrain,featTrain);
            funLL = @(theta)algs(a).isingFn(theta, ss, L, N, featTrain);

            lambdaL2 = ones(size(theta))*lambdaBar;
            [thetaML,~, ~, outputInfo] = minFunc(@penalizedL2, theta, options, funLL, lambdaL2);
            llTrace(1:length(outputInfo.trace.fval), a) = outputInfo.trace.fval;
%           [thetaML,llTrace(:,a)] = L1General2_TMP(funLL, theta, lambdaL1,options);
        else
            thetaML=mean(objectsPresentTrain,2);
            numer=bsxfun(@times,thetaML,objectsPresentTrain);
            denom = log(exp(numer)+1);
            llTrace(1:options.maxIter,a) = -(sum(numer(:)-denom(:)));
        end
        
        aucAll(:,:,a) = mainInfer(featTest,objectsPresentTest,algs(a).inferFn,thetaML);
    end
end
