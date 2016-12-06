function [ ll,grad ] = getLlikCRFMean(theta, ss, L, N, feats, seqlen)
    % [ll,grad] = getLlikCRFMean(theta, ss, L, N, feats)
    % Compute the negative log-likelihood and gradient for the CRF with a
    % mean-field approximation.
    % 
    % Inputs:
    %    theta: Current estimate of the parameters. The
    %           first 4 entries, theta(1:4) are the parameters governing the
    %           tertiary potentials. theta(5:end) govern the interaction
    %           with the image features.
    %    ss: Sufficient statistics needed to compute log-likelihood and
    %        gradients. These are returned by suffStatsCRF.m
    %    L: Number of training examples
    %    N: Length L vector of number of possible edges for each training 
    %       example 
    %    feats: M x L vector of image features, where M is the number of
    %           features per image.  This is not necessarily used in this
    %           function (but we provide it so calls to MRFs and CRFs are
    %           identical).
    % Outputs:
    %    ll: the data log-likelihood
    %    grad: the gradient of the log-likelihood function for the given value
    %          of theta


    % separate out the sufficient statistics relating to the
    % node-occurence statistics, and the node-image statistics. why???
%     featSS = ss(N*(N+1)/2+1:end);
%     ss = ss(1:N*(N+1)/2);
    
    % Compute joint and marginal statistics for current model
    mus = margProbMean(theta,N,feats,seqlen);
    % DEBUG!!
%     mus = cell(L, 1);
%     for l = 1:L
%         mus{l} = ones(N(l), 1);
%     end
    gamma = theta(5:end-3);
    keyboard
    F = 0;
    for l = 1:L
        F = F + calcF(mus{l}, feats{l}, seqlen(l), ...
                        theta(1:4), gamma, theta(end-2), theta(end-1), ...
                        theta(end));
    end
                
    ll = theta'*ss - F;
    keyboard
    
    phi_mu = zeros(size(ss));
    for l = 1:L
        phi_mu(1:N) = phi_mu(1:N) + mus(1:N,l);
        edgeIndex = N+1;
        for ii = 1:N-1
            for jj = ii+1:N
                phi_mu(edgeIndex) = phi_mu(edgeIndex) + mus(ii,l)*mus(jj,l);
                edgeIndex = edgeIndex + 1;
            end
        end
    end
    temp = mus';
    temp = reshape(temp, [1, size(temp)]);
    phi_feat = sum(bsxfun(@times,temp,feats),2);
    phi_mu(edgeIndex:end) = phi_feat(:);
    grad = ss - phi_mu;
    
    ll = -ll;
    grad = -grad;
    
    
end



