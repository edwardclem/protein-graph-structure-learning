function [mus] = margProbMean(theta,feats)
% mus = margProbMean(theta,N,feats)
% Computes the variational paramteres for the mean-field equations, given
% model parameters.
%
% Inputs:
%    theta: Parameters of the model. theta(1:N) encodes the parameters governing
%           the unary potentials of the model (\theta_s in the handout).
%           theta(N+1:end) encodes the parameters governing the pairwise
%           potentials  (\theta_{st}) in the handout. N is the number of
%           variable nodes in the model. N is the number of variable nodes
%           in the model.
%    thetaMat: an N x N matrix, where thetaMat(i,j) is the paramter
%              governing the interaction between node i and j. In the
%              notation of the handout, this is \theta_{i,j}
%    feats: Set of image features of size [#features,#examples], if this
%           model uses image features (eg, CRF). If image features are not
%           used, this argument must be empty, [].
%    partialLabels: a partial labelling of the image. Each entry is 0/1 to
%                   indicate absence/presence of the category, or -1 if it
%                   is unknown.  By default, assume all labels are unknown.
% Outputs:
%   mus: An N x #examples matrix containing the variational parameters for
%        the mean-field approximation. In the case of approximating an MRF,
%        #examples = 1, and mus is a column vector.

    M_ITER = 20;
    sumGam = feats'*theta(5:end);
    mus = 0.5*ones(

    if(isempty(feats))
        nEx = 1;
        sumGam = zeros(1,N);
    else
        nEx = size(feats,2);
        gam = reshape(theta(N*(N+1)/2+1:end),size(feats,1),N);
        sumGam = feats'*gam; % Should be useful for CRF mean field updates
    end
    if(isempty(partialLabels))
        indSweep = 1:N;
        N_sweep = N;
        mus = 0.5*ones(N_sweep,nEx);
    else
        indSweep = find(partialLabels==-1);
        N_sweep = numel(indSweep);
        mus = partialLabels;
        mus(indSweep) = 0.5;
    end

    for(mm=1:M_ITER)
        for(i=1:N_sweep)
            n = indSweep(i);
            % Update mus(n,:) given fixed values for other rows
            alpha = exp(thetaMat(n,:)*mus + theta(n) + sumGam(:,n)');
            mus(n,:) = alpha./(1 + alpha);
        end
    end

end

