function [completeLabel] = doInferMean(partialLabel,theta,feats)
%  [completeLabel] = doInferMean(partialLabel,theta,feats)
%  Given a partial labelling (presence/absence of objects) of a scene,
%  returns a completed labelling.
%  
%  Inputs:
%     partialLabel: an N x 1 vector indicating a partial labelling of the
%                   image. Its entries have the following possibilities:
%                      0: absence of object
%                      1: presence of object
%                     -1: label unavailable. Must be inferred. 
%     theta: Vector of parameters governing this model.
%     feats: M x 1 vector of image features. If features are not needed for
%            inference, feats should be empty, [].
%  Outputs:
%     completeLabel: A probabilistic labelling of the image. If
%                    partialLabel(i) == -1, then completeLabel(i) is the
%                    probability of the i-th objet being present in the 
%                    image. Otherwise, completeLabel(i) = partialLabel(i).
    N = numel(partialLabel);
    thetaMat = zeros(N,N);
    idx = N+1;
    for(n=1:N)
        ext = theta(idx:idx+(N-n)-1);
        thetaMat(n,n+1:end) = ext;
        idx = idx+N-n;
    end
    thetaMat=(thetaMat'+thetaMat);

    completeLabel = margProbMean(theta,thetaMat,feats,partialLabel);   
end

