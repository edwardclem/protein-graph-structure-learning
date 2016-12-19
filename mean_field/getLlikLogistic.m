function [ ll, grad ] = getLlikLogistic(theta, gt, L, feats, seqlen, crfOpt)
    % [ll,grad] = getLlikLogistic(theta, ss, L, N, feats)
    % Compute the negative log-likelihood and gradient for a logistic
    % regression model
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
    
    % Compute cost and gradient for current model
    ll = 0;
    grad = zeros(size(theta));
    if crfOpt.verbose; fprintf('\tCalculating cost and gradient... '); end;
    tstart = tic;
    gamma = theta(1:end-3);
    for l = 1:L
        [cost_l, grad_l] = calcLogistic(int32(gt{l}), feats{l}, seqlen(l), ...
                            theta(1:4), gamma, theta(end-2), ...
                            theta(end-1), theta(end), crfOpt.condDist);
        ll = ll + cost_l;
        grad = grad + grad_l;
    end
    tstop = toc(tstart);
    if crfOpt.verbose; fprintf('done. Time: %0.1fs. GradVal: %0.3f\n', tstop, norm(grad)); end;
    
    ll = -ll;
    grad = -grad;
    
end

function prob = sigmoid(alpha)
prob = 1./(1 + exp(-alpha));
end



