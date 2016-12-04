function [ll,grad] = getLlikLogistic(theta, ss, L, N, feats)
    % [ll,grad] = getLlikLogistic(theta, ss, L, N, feats)
    % Compute the negative log-likelihood and gradient for the CRF
    % 
    % Inputs:
    %    theta: Current estimate of the parameters of the Ising model. The
    %           first N entries, theta(1:N) are the parameters governing the
    %           unary potentials. theta(N+1:end) govern the interaction
    %           with the image features.
    %    ss: Sufficient statistics needed to compute log-likelihood and
    %        gradients. These are returned by suffStatsLogistic.m
    %    L: Number of training examples
    %    N: Number of unary-terms in theta (number of object categories)
    %    feats: M x L vector of image features, where M is the number of
    %           features per image.

    % Outputs:
    %    ll: the log-likelihood of the model
    %    grad: the gradient of the log-likelihood function for the given value
    %          of theta


    % Compute joint and marginal statistics for current model

    featSS= ss(N+1:end);
    ss = ss(1:N);
    
    [log_Pjoint,log_normJoint] =unnormProbLogistic(theta, N, feats);
    Pmarg = exp(log_normJoint);  

    grad(1:N,1) = (ss-sum(Pmarg,1)');
    temp = feats*Pmarg; temp = temp(:);
    grad(N+1:N+size(feats,1)*N) = [featSS-temp];
    
    unary = theta(1:N)'*ss;
    featProb =  theta(N+1:end)'*featSS;
    
    temp = exp(log_Pjoint);
    temp = log(temp+1);
    totLogPart = sum(temp(:));
    
    ll = unary + featProb - totLogPart;
    
    % Invert signs for minimization objective
    ll = -ll;
    grad = -grad;

end



