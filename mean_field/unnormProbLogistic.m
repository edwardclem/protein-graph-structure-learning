function [log_Pjoint,log_normJoint] = unnormProbLogistic(theta, N,feats)
% P = unnormProb(theta,N)
% Computes the unnormalized log probabilities of all joint states
% 
% Inputs:
%    theta: Parameters of the model. theta(1:N) encodes the parameters governing
%           the unary potentials of the model (\theta_s in the handout).
%           theta(N+1:end) encodes the parameters governing the pairwise
%           potentials  (\theta_{st}) in the handout.
%    N: The number of variables nodes in the model.
% Outputs:
%   log_Pjoint: The log of the joint probability of each of the
%               2^N configurations of the (binary) variable nodes for each
%               example.

    gam = reshape(theta(N+1:end),size(feats,1),N);
    thetaS = theta(1:N);
    
    sumGam = feats'*gam;
    
    pxs = exp(bsxfun(@plus,thetaS',sumGam));
    
    log_Pjoint = log(pxs);
    log_normJoint = log(pxs./(1+pxs));
end




