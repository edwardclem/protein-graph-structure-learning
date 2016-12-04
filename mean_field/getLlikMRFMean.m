function [ll,grad] = getLlikMRFMean(theta, ss, L, N, feats)
    % [ll,grad] = getLlikMRFMean(theta, ss, L, N, feats)
    % Compute the negative log-likelihood and gradient for the MRF with a
    % mean-field approximation.
    % 
    % Inputs:
    %    theta: Current estimate of the parameters of the Ising model. The
    %           first N entries, theta(1:N) are the parameters governing the
    %           unary potentials. theta(N+1:N*(N+1)/2) govern the pairwise
    %           potentials.
    %    ss: Sufficient statistics needed to compute log-likelihood and
    %        gradients. These are returned by suffStatsMRF.m
    %    L: Number of training examples
    %    N: Number of unary-terms in theta (number of object categories)
    %    feats: M x L vector of image features, where M is the number of
    %           features per image. This is not necessarily used in this
    %           function (but we provide it so calls to MRFs and CRFs are
    %           identical).
    % Outputs:
    %    ll: the data log-likelihood
    %    grad: the gradient of the log-likelihood function for the given value
    %          of theta

    % Compute joint and marginal statistics for current model
    
    % extract the parts of the parameter vector that deal with pairwise
    % interactions between the nodes. In the notation of the handout,
    % thetaMat(s,t) = \theta_{s,t}
    thetaMat = zeros(N,N);
    idx = N+1;
    for(n=1:N)
        ext = theta(idx:idx+(N-n)-1);
        thetaMat(n,n+1:end) = ext;
        idx = idx+N-n;
    end
    thetaMat=(thetaMat'+thetaMat);
    mus = margProbMean(theta,thetaMat,[],[]);
    
    % FILL THIS IN!
    F = mus'*thetaMat*mus/2 + theta(1:N)'*mus - mus'*log(mus) - (1 - mus)'*log(1 - mus);
    ll = theta'*ss - L*F;
    phi_mu = zeros(size(ss));
    phi_mu(1:N) = mus;
    edgeIndex = N+1;
    for ii = 1:N-1
        for jj = ii+1:N
            phi_mu(edgeIndex) = mus(ii)*mus(jj);
            edgeIndex = edgeIndex + 1;
        end
    end
    grad = ss - L*phi_mu;
    
    grad = -grad;
    ll = -ll;
end

