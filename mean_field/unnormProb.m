function log_Pjoint = unnormProb(theta,feats,partialLabels)
% P = unnormProb(theta,N,feats)
% Computes the unnormalized log probabilities of all joint states
% 
% Inputs:
%    theta: Parameters of the model. theta(1:N) encodes the parameters governing
%           the unary potentials of the model (\theta_s in the handout).
%           theta(N+1:end) encodes the parameters governing the pairwise
%           potentials  (\theta_{st}) in the handout. N is the number of
%           variable nodes in the model.
%    feats: Set of image features of size [#features,#examples], if this
%           model uses image features (eg, CRF). If image features are not
%           used, this argument must be empty, [].
%    partialLabels: a partial labelling of the image. Each entry is 0/1 to
%                   indicate absence/presence of the category, or -1 if it
%                   is unknown (eg, must be summed over).
% Outputs:
%   log_Pjoint: The log of the joint probability of each of the
%               2^N_sweep configurations of the (binary) variable nodes for
%               each example. Here, N_sweep is the number of variable nodes
%               that were unspecified in partialLabels. The order of the
%               states is given in the order given in partialLabels.

    N = numel(partialLabels);
    if(isempty(feats))
        nEx = 1;
        sumGam = zeros(1,N);
    else    
        nEx = size(feats,2);
        gam = reshape(theta(N*(N+1)/2+1:end),size(feats,1),N);
        sumGam = feats'*gam; %[nEx,N]
    end

    indSweep = find(partialLabels==-1);
    N_sweep = numel(indSweep);
    
    log_Pjoint = zeros(2^N_sweep,nEx);
    state = zeros(N_sweep,1);

    stateComplete = partialLabels;
    for st=1:2^N_sweep
        stateComplete(indSweep) = state;
        
        %unary
        unary = theta(1:N)' * stateComplete;
        
        % Pairwise potentials
        edgeIndex = N+1;
        pair = 0;
        for ii = 1:N-1
            for jj = ii+1:N
                pair =pair + theta(edgeIndex)*stateComplete(ii)*stateComplete(jj);
                edgeIndex = edgeIndex + 1;
            end
        end

        % feature potentials
        featPot = sumGam*stateComplete;
        log_Pjoint(st,:) = featPot + unary + pair;
        
        state(1) = state(1) + 1;
        for (i = 1:N_sweep-1)
            if (state(i) == 2)
                state(i+1) = state(i+1) + 1;
                state(i) = 0;
            else
                break;
            end
        end
    end
end
