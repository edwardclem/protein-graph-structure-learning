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
%         mus{l} = 0.5*ones(N(l), 1);
%     end
    gamma = theta(5:end-3);
    F = 0;
    gradF = zeros(size(ss));
    fprintf('\tCalculating F and gradF... ');
    tstart = tic;
    for l = 1:L
        [F_l, gradF_l] = calcF(mus{l}, feats{l}, seqlen(l), ...
                            theta(1:4), gamma, theta(end-2), ...
                            theta(end-1), theta(end));
        F = F + F_l;
        gradF = gradF + gradF_l;
    end
    tstop = toc(tstart);
    fprintf('done. Time: %0.1fs.\n', tstop);
    ll = theta'*ss - F;
    grad = ss - gradF;
    
    fprintf('\t\t||gradient|| = %0.3f\n', norm(grad))
    
    ll = -ll;
    grad = -grad;
    
    
end



