function Pmarg = margProbCRF(Pjoint, N, feats)
% Pmarg = margProbCRF(Pjoint,N)
% Computes expected values of the model needed for gradient descent. 
% 
% Inputs:
%   N: The number of variables nodes in the model.
%   Pjoint: The joint probability of each of the 2^N configurations of the
%           (binary) variable nodes. This input should be computed using
%   >> Pjoint = exp(unnormProb(thetaJoint,N));
%   >> Pjoint = Pjoint / sum(Pjoint);
%   feats: a [#Features x #examples] matrix of image features.
% Outputs:
%   Pmarg: Expected values necessary for gradient descent. Entries...
%          1) Pmarg(1:N) deal with expected values for the unary terms
%          2) Pmarg(N+1:N*(N+1)/2 deal with expected values for the
%             pairwise terms (x_s * x_t), 
%          3) the rest of Pmarg deal with execpted values between the nodes
%             and the image features (ie, x_s feats_k).

    nFeats = size(feats,1);
    Pmarg = zeros(N*(N+1)/2+nFeats*N,1);
    
    % expectations under model for parameters not involving features are
    % the same as the MRF
    Pmarg(1:N*(N+1)/2) = margProb(sum(Pjoint,2),N);
    
    Pmarg_feat = zeros(nFeats,N);
    featMargs = feats*Pjoint';

    state = zeros(1,N);
    for m = 1:2^N
        
      Pmarg_feat(:,state==1) = bsxfun(@plus, Pmarg_feat(:,state==1), featMargs(:,m));
      state(1) = state(1) + 1;
        for (i = 1:N-1)
            if (state(i) == 2)
                state(i+1) = state(i+1) + 1;
                state(i) = 0;
            else
                break;
            end
        end
    end
    Pmarg(N*(N+1)/2+1:end) = Pmarg_feat(:);
end
