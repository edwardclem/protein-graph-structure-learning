function Pmarg = margProb(Pjoint, N)
% Pmarg = margProb(Pjoint,N)
% Computes the marginal probabilities of Ising sufficient stats
%
% Inputs:
%   N: The number of variables nodes in the model.
%   Pjoint: The joint probability of each of the 2^N configurations of the
%           (binary) variable nodes. This input should be computed using
%   >> Pjoint = exp(unnormProb(thetaJoint,N));
%   >> Pjoint = Pjoint / sum(Pjoint);
% Outputs:
%   Pmarg: The marginal probabilities of Ising sufficient stats

    Pmarg = zeros(N*(N+1)/2,1);

    state = zeros(1,N);
    for m = 1:2^N
        % Single-node mean probabilities
        Pmarg(state==1) = Pmarg(state==1)+Pjoint(m);

        % Pairwise co-occurrence probabilities
        edgeIndex = N+1;
        for ii = 1:N-1
            for jj = ii+1:N
                Pmarg(edgeIndex) = Pmarg(edgeIndex) + Pjoint(m)*state(ii)*state(jj);
                edgeIndex = edgeIndex + 1;
            end
        end

        state(1) = state(1) + 1;
        for (i = 1:N-1)
            if (state(i) == 2)
                state(i+1) = state(i+1) + 1;
                state(i) = 0;
            else
                break;
            end
        end
    end
end
%

