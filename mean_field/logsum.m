function res = logsum(x,dim)
% Computes the log of a sum, where we have the log of each term of the sum.
% Eg, computes \log(\sum_i a_i) when we have log(a_i) \forall i.
% Does this in a way that avoids numerical issues.
    if(nargin < 2)
        dim = 1;
    end

    m = max(x,[],dim);
    x = bsxfun(@minus,x,m);
    res = m + log(sum(exp(x),dim));
end

