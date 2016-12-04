function [ll,grad] = getLlikMRF(theta, ss, L, N, feats)
    % [ll,grad] = getLlikMRF(theta, ss, L, N, feats)
    % Compute the negative log-likelihood and gradient for fully-connected Ising model
    % 
    % Inputs:
    %    theta: Current estimate of the parameters of the Ising model. The
    %           first N entries, theta(1:N) are the parameters governing the
    %           unary potentials, and the rest govern the pairwise potentials
    %           of the Ising model.
    %    ss: Sufficient statistics needed to compute log-likelihood and
    %        gradients. These are returned by suffStatsMRF.m
    %    L: Number of training examples
    %    N: Number of unary-terms in theta (number of object categories)
    %    feats: M x L vector of image features, where M is the number of
    %           features per image.

    % Outputs:
    %    ll: the log-likelihood of the model
    %    grad: the gradient of the log-likelihood function for the given value
    %          of theta


    % Although we are given the image features, feats, the MRF does not
    % make use of them. We are passed it in the method signature just so
    % the CRF and MRF function signatures are identical.
    log_Pjoint = unnormProb(theta, [], -ones(N,1));
    logPart = logsum(log_Pjoint,1);
    Pjoint = exp(bsxfun(@minus, log_Pjoint,logPart));
    Pmarg = margProb(Pjoint, N);
    
    % Compute log-likelihood of data under current model
    ll = theta'*ss - L*logPart;
    
    % Compute gradient of log-likelihood
    grad = ss - L*Pmarg;

    % Invert signs for minimization objective
    ll = -ll;
    grad = -grad;
end

