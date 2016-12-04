function [completeLabel] = doInfer(partialLabel,theta,feats)
%  [completeLabel] = doInfer(partialLabel,theta,feats)
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

    indSweep = find(partialLabel==-1);
    N_sweep = numel(indSweep);   
    
    log_Pjoint = unnormProb(theta,feats,partialLabel);

    joint = exp(log_Pjoint-logsum(log_Pjoint,1));
    
    % enumerate the categories we weren't given
    margs = zeros(N_sweep,1);
    state = zeros(N_sweep,1);
    for(i=1:2^N_sweep)
        margs = margs + joint(i)*state;
        
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
    completeLabel = partialLabel;
    completeLabel(indSweep) = margs;
    
end

