function [completeLabel] = crfInfer(partialLabel,theta,feats)
%  [completeLabel] = crfInfer(partialLabel,theta,feats)
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

    completeLabel = doInfer(partialLabel,theta,feats);
    
end

