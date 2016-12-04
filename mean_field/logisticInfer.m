function [completeLabel] = logisticInfer(partialLabel,theta,feats)
%  [completeLabel] = logisticInfer(partialLabel,theta,feats)
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
%     feats: M x 1 vector of image features.
%  Outputs:
%     completeLabel: A probabilistic labelling of the image. If
%                    partialLabel(i) == -1, then completeLabel(i) is the
%                    probability of the i-th objet being present in the 
%                    image. Otherwise, completeLabel(i) = partialLabel(i).

    completeLabel = partialLabel;

    missing = find(partialLabel==-1);
    N = numel(partialLabel);
    
    gam = reshape(theta(N+1:end),size(feats,1),N);
    
    for(i=1:numel(missing))
        ind = missing(i);
        
        unary = theta(ind);
        featprob = gam(:,ind)'*feats;
        numer = exp(unary+featprob);
        
        completeLabel(ind) = numer./(numer+1);
        
    end
end

