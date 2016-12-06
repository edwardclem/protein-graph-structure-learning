function [mus] = margProbMean(theta,N,feats,seqlen)
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

    L = numel(feats);
    mus = cell(L, 1);
    gamma = theta(5:end-3);
    tstart = tic;
    reverseStr = '';
    fprintf('\tCalculating muhat. ');
    for l = 1:L
        mus{l} = calc_muhat(uint32(N(l)), feats{l}, seqlen(l), ...
                        theta(1:4), gamma, theta(end-2), theta(end-1), ...
                        theta(end));
        percentDone = 100 * l / L;
        msg = sprintf('Percent done: %3.1f', percentDone);
        fprintf([reverseStr, msg]);
        reverseStr = repmat(sprintf('\b'), 1, length(msg));
    end
    tstop = toc(tstart);
    fprintf('. Time: %0.1fs.\n', tstop);
end

