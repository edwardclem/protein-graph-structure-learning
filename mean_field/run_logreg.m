function [ output_args ] = run_logreg(train_data, test_data, outfile)

seed = 0;
rng(seed);

% Load training data
[ss_proteins, features_aa, seqlen_all, gt] = load_data(train_data);
L = numel(features_aa); % seqlen variable

% Run Logistic Regression

% Options
lambdaBar = 0;
options.maxIter = 1000;
options.progTol = 1e-9;
crfOpt.verbose = 0; % Print things while running? 
crfOpt.nThreads = 4; % Number of threads to use
crfOpt.condDist = 0; % condition on edges with sequence distance less than this

% Setup inputs
%log likelihood is indeed fun
funLL = @(theta)getLlikLogistic(theta, gt, L, features_aa, seqlen_all, crfOpt);
theta = zeros([numel(ss_proteins) - 4, 1]);
lambdaL2 = ones(size(theta))*lambdaBar;
llTrace = NaN(options.maxIter, 1);

% Run Mean Field
fprintf('Starting Logistic Regression CRF\n');
tstart = tic;
[thetaML,~, ~, outputInfo] = minFunc(@penalizedL2, theta, options, funLL, lambdaL2);

tstop = toc(tstart);
fprintf('Gradient Descent Elapsed in %0.1fs.\n', tstop);
llTrace(1:length(outputInfo.trace.fval)) = outputInfo.trace.fval;

[~, feats_test, seqlen_test, gt_test] = load_data(test_data);
L_test = numel(feats_test); % seqlen variables
%thetaTest = [thetaML(end); thetaML(1:end-1)];
scores = cell(L_test, 1);
fprintf('calculating performance on test data');
for l = 1:L_test
    scores{l} = logInfer(feats_test{l}, seqlen_test(l), thetaML(1:end-3), ...
                        thetaML(end-2), thetaML(end-1), thetaML(end), crfOpt.condDist);
end
all_gt = double(vertcat(gt{:}));
scores = vertcat(scores{:});
[X, Y, T, AUC] = perfcurve(all_gt, scores, 1);
fprintf('Saving to output file');
save(outfile, 'X', 'Y', 'T', 'AUC', 'thetaML', 'llTrace');
end

