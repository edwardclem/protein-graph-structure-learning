function [ll,grad] = getLlikCRF(theta, ss, L, N, feats)
    % [ll,grad] = getLlikCRF(theta, ss, L, N, feats)
    % Compute the negative log-likelihood and gradient for the CRF
    % 
    % Inputs:
    %    theta: Current estimate of the parameters of the Ising model. The
    %           first N entries, theta(1:N) are the parameters governing the
    %           unary potentials. theta(N+1:N*(N+1)/2) govern the pairwise
    %           potentials. theta(N*(N+1)/2+1:end) govern the interaction
    %           with the image features.
    %    ss: Sufficient statistics needed to compute log-likelihood and
    %        gradients. These are returned by suffStatsCRF.m
    %    L: Number of training examples
    %    N: Number of unary-terms in theta (number of object categories)
    %    feats: M x L vector of image features, where M is the number of
    %           features per image.

    % Outputs:
    %    ll: the data log-likelihood
    %    grad: the gradient of the log-likelihood function for the given value
    %          of theta

    % Compute joint and marginal statistics for current model
    log_Pjoint = unnormProb(theta, feats, -ones(N,1));
    logPart = logsum(log_Pjoint,1);
    Pjoint = exp(bsxfun(@minus, log_Pjoint,logPart));
    Pmarg = margProbCRF(Pjoint, N, feats);
    
    ll = theta'*ss - sum(logPart);
    
    grad = ss - Pmarg;
    
    ll = -ll;
    grad = -grad;
    
end



