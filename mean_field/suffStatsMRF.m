function ss = suffStatsMRF(objectsPresent,feats)
% ss = suffStatsMRF(data,featTrain)
    % Compute the sufficient statistics necessary for computing the
    % gradient of an MRF. Assume there are N object classes and L training
    % examples. Each image has M features.
    %
    % Inputs:
    %   objectsPresent: N x L matrix of 0/1 labels indicating the
    %                   absence/presence of an object in an example.
    %   feats: M x L matrix of image features. May or may not be
    %          necessary input for this function.
    % Outputs:
    %   ss: Vector of sufficient statistics.

    % Compute the sufficient statistics for Ising pairwise MRF learning
    N  = size(objectsPresent,1);
    ss = zeros(N*(N+1)/2,1);

    % Unary statistics
    ss(1:N) = sum(objectsPresent,2);

    % Pairwise statistics
    edgeIndex = N+1;
    for ii = 1:N-1
        for jj = ii+1:N
            ss(edgeIndex) = sum(objectsPresent(ii,:).*objectsPresent(jj,:));
            edgeIndex = edgeIndex + 1;
        end
    end
end


