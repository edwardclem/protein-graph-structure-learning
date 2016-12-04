function ss = suffStatsLogistic(objectsPresent, feats)
% ss = suffStatsMRF(data,featTrain)
    % Compute the sufficient statistics necessary for computing the
    % gradient of an MRF. Assume there are N object classes and L training
    % examples. Each image has M features.
    %
    % Inputs:
    %   objectsPresent: N x L matrix of 0/1 labels indicating the
    %                   absence/presence of an object in an example.
    %   featTrain: M x L matrix of image features. May or may not be
    %              necessary input for this function.
    % Outputs:
    %   ss: Vector of sufficient statistics.

    N  = size(objectsPresent,1);

    % Unary statistics
    ss(1:N,1) = sum(objectsPresent,2);
    
    temp = objectsPresent';
    temp = reshape(temp,[1,size(temp)]);
    featSS = squeeze(sum(bsxfun(@times,temp,feats),2));

    ss= [ss; featSS(:)];
end

