function [completeLabel] = baseInfer(partialLabel,theta,feats)
    indSweep = find(partialLabel==-1);
    completeLabel = partialLabel;
    completeLabel(indSweep) = theta(indSweep);
end

