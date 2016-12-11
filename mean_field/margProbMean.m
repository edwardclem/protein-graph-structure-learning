function [mus] = margProbMean(theta,N,feats,seqlen,crfOpt)
% mus = margProbMean(theta,N,feats)
% Computes the variational paramteres for the mean-field equations, given
% model parameters.
%
% Inputs:
%    theta: Parameters of the model. theta(1:4) encodes the parameters 
%           governing the tertiary potentials. theta(5:end-3) encode the
%           parameters governing the amino acid potentials. theta(end-2)
%           encodes the distance parameter, theta(end-1) encodes the seqlen
%           parameter, and theta(end) encodes the prior.
%
%    N:     a length L vector containing the number of edges in the l-th
%           example.
%
%    feats: Set of protein features. A L length cell array, with each array
%           feats(l) containing a [#feats x N(l)] array.
% Outputs:
%   mus: An length L cell array containing the variational parameters for
%        the mean-field approximation. Each sub array mus(l) contains N(l)
%        parameters.
    if nargin < 4
        crfOpt.verbose = 0;
        crfOpt.nThreads = uint32(1);
    end
    if ~isa(crfOpt.nThreads, 'uint32')
        crfOpt.nThreads = uint32(crfOpt.nThreads);
    end
    if ~isa(crfOpt.condDist, 'uint32')
        crfOpt.condDist = uint32(crfOpt.condDist);
    end

    L = numel(feats);
    gamma = theta(5:end-3);
    tstart = tic;
    if crfOpt.verbose; fprintf('\tCalculating muhat... '); end;
    if (crfOpt.nThreads == 1)
        mus = calc_muhat(N, feats, seqlen, theta(1:4), gamma, ...
                        theta(end-2), theta(end-1), theta(end), uint32(L));
    else
%         if (crfOpt.condDist > 0)
            mus = calc_muhat_parallel_cond(N, feats, seqlen, theta(1:4), gamma, ...
                            theta(end-2), theta(end-1), theta(end), uint32(L), ...
                            crfOpt.nThreads, crfOpt.condDist);
%         else
%             mus = calc_muhat_parallel(N, feats, seqlen, theta(1:4), gamma, ...
%                             theta(end-2), theta(end-1), theta(end), uint32(L), ...
%                             crfOpt.nThreads);
%         end
    end
    tstop = toc(tstart);
    if crfOpt.verbose; fprintf('. Time: %0.1fs.\n', tstop); end;
end

