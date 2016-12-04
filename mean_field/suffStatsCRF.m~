function [ss, feats] = suffStatsCRF(ss_proteins)
% ss = suffStatsCRF(data,featTrain)
    % Compute the sufficient statistics necessary for computing the
    % gradient of a CRF. Assume L training examples. Each protein has M
    % features. Assume last four features are non-conditional features.
    %
    % Inputs:
    %   ss_proteins: M x L matrix of protein statistics. 
    % Outputs:
    %   ss: Vector of sufficient statistics.
    %   feats: (M - 4) x L matrix of protein features.

    % TODO: Deal with seqlen variable
    nStatistics = size(ss_proteins,1);
    
    feats = ss_proteins(1:end-4,:);
    ss = sum(ss_proteins, 1)';
    ss = [ss(end-3:end) ss(1:end-4)]; % Switch order so that conditional stats second

end


