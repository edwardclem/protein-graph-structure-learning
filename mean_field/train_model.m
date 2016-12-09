function [] = train_model(train_data, test_data, outfile, nThreads)

seed = 0;
rng(seed);

%% Load data
disp(train_data);
[ss_proteins, features_aa, seqlen_all, ~] = load_data(train_data);
L = numel(features_aa); % seqlen variable
disp(L);
N = seqlen_all.*(seqlen_all - 1)/2; % Number of possible edges

%% Run Mean Field CRF

% Options
lambdaBar = 0;
options.maxIter=1000;
options.progTol=1e-11;
crfOpt.verbose=0;
crfOpt.nThreads = nThreads; % Number of threads to use

% Setup inputs
funLL = @(theta)getLlikCRFMean(theta, ss_proteins, L, N, features_aa, seqlen_all, crfOpt);
theta = zeros([numel(ss_proteins), 1]);
lambdaL2 = ones(size(theta))*lambdaBar;
llTrace = NaN(options.maxIter, 1);


% Run Mean Field
[thetaML,~, ~, outputInfo] = minFunc(@penalizedL2, theta, options, funLL, lambdaL2);
llTrace(1:length(outputInfo.trace.fval)) = outputInfo.trace.fval;

%%load testing data
[~, features_test, seqlen_test, gt] = load_data(test_data);
N_test = seqlen_test.*(seqlen_test - 1)/2; % Number of possible edges

%% Plot results
muhat = margProbMean(thetaML, N, features_test, seqlen_test, crfOpt); % change to test data
t_val = 1:-0.001:0.001;
FAR = zeros(size(t_val));
DR = zeros(size(t_val));
for l = 1:L
    DR = DR + arrayfun(@(t) nnz((muhat{l} > t) & (gt{l} == 1))/nnz(gt{l} == 1), t_val);
    FAR = FAR + arrayfun(@(t) nnz((muhat{l} > t) & (gt{l} == 0))/nnz(gt{l} == 0), t_val);
end
DR = DR/L;
FAR = FAR/L;
save(outfile, 'DR', 'FAR', 'thetaML')

end
