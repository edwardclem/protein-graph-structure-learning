function [ ll,grad ] = getLlikCRFPll(theta, gt, ss, L, N, feats, seqlen, crfOpt)
    % [ll,grad] = getLlikCRFMean(theta, ss, L, N, feats)
    % Compute the negative log-likelihood and gradient for the CRF with a
    % mean-field approximation.
    % 
    % Inputs:
    %    theta: Parameters of the model. theta(1:4) encodes the parameters 
    %       governing the tertiary potentials. theta(5:end-3) encode the
    %       parameters governing the amino acid potentials. theta(end-2)
    %       encodes the distance parameter, theta(end-1) encodes the seqlen
    %       parameter, and theta(end) encodes the prior.
    %    N:  a length L vector containing the number of edges in the l-th
    %        example.
    %    feats: Set of protein features. A L length cell array, with each array
    %           feats(l) containing a [#feats x N(l)] array.
    %    ss: Sufficient statistics needed to compute log-likelihood and
    %        gradients. These are returned by load_data.m
    %    L: Number of training examples
    %    crfOpt: Options for things...
    % Outputs:
    %    ll: the data log-likelihood
    %    grad: the gradient of the log-likelihood function for the given value
    %          of theta
    
    if ~isfield(crfOpt, 'verbose')
        crfOpt.verbose = 0;
    end
    if ~isfield(crfOpt, 'nThreads')
        crfOpt.nThreads = int32(1);
    end
    if ~isa(crfOpt.nThreads, 'int32')
        crfOpt.nThreads = int32(crfOpt.nThreads);
    end
    if ~isa(crfOpt.condDist, 'int32')
        crfOpt.condDist = int32(crfOpt.condDist);
    end
    
    
    % Compute joint and marginal statistics for current model
    gamma = theta(5:end-3);
    Pll = 0;
    gradPll = zeros(size(ss));
    if crfOpt.verbose; fprintf('\tCalculating Pll and gradPll... '); end;
    tstart = tic;
    for l = 1:L
        [Pll_l, gradPll_l] = calcPll(gt{l}, feats{l}, seqlen(l), ...
                            theta(1:4), gamma, theta(end-2), ...
                            theta(end-1), theta(end), crfOpt.condDist);
        Pll = Pll + Pll_l;
        gradPll = gradPll + gradPll_l;
    end
    tstop = toc(tstart);
    %ll = theta'*ss - Pll;
    ll = Pll;
    %grad = ss - gradPll;
    grad = gradPll;
    %grad(1:2) = 0;
%     disp([ss(3:4) gradPll(3:4)])
    %keyboard
    if crfOpt.verbose; fprintf('done. Time: %0.1fs. GradVal: %0.3f\n', tstop, norm(grad)); end;
    
    ll = -ll;
    grad = -grad;
    % DEBUG!!
    %grad(1:4) = 0;
    %grad(end-2) = 0;
    
end



