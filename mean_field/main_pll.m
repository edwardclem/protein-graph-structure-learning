seed = 0;
rng(seed);

%% Load data, split into train + test
directory = 'data/data_parallel';
[ss_proteins, features_aa, seqlen_all, gt] = load_data(directory);
L = numel(features_aa); % seqlen variable
N = seqlen_all.*(seqlen_all - 1)/2; % Number of possible edges

%% Run Mean Field CRF

% Options
lambdaBar = 0;
options.maxIter = 1000;
options.progTol = 1e-9;
crfOpt.verbose = 0; % Print things while running? 
crfOpt.nThreads = 4; % Number of threads to use
crfOpt.condDist = 0; % condition on edges with sequence distance less than this

% Setup inputs
%log likelihood is indeed fun
funLL = @(theta)getLlikCRFPll(theta, gt, ss_proteins, L, N, features_aa, seqlen_all, crfOpt);
theta = zeros([numel(ss_proteins), 1]);
lambdaL2 = ones(size(theta))*lambdaBar;
llTrace = NaN(options.maxIter, 1);

% Run Mean Field
fprintf('Starting Gradient Descent Pseudologlikelihood CRF\n');
tstart = tic;
[thetaML,~, ~, outputInfo] = minFunc(@penalizedL2, theta, options, funLL, lambdaL2);
tstop = toc(tstart);
fprintf('Gradient Descent Elapsed in %0.1fs.\n', tstop);
llTrace(1:length(outputInfo.trace.fval)) = outputInfo.trace.fval;

%% Plot results
muhat = margProbMean(thetaML, N, features_aa, seqlen_all, crfOpt); % change to test data

%%
t_val = 1:-0.001:0.001;

all_mus = vertcat(muhat{1:end});
all_gt = vertcat(gt{1:end});

[X, Y, T, AUC] = perfcurve(all_gt,  all_mus, 1);
figure;
hold on
plot(X, Y);
plot(0:0.1:1, 0:0.1:1);