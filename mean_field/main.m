seed = 0;
rng(seed);

%% Load data, split into train + test
directory = '../data/summed_suffstats/';
[ss_proteins, features_aa, seqlen_all, gt] = load_data(directory);
L = numel(features_aa); % seqlen variable
N = seqlen_all.*(seqlen_all - 1)/2; % Number of possible edges

%% Run Mean Field CRF

% Options
lambdaBar = 0.01;
options.maxIter=1000;

% Setup inputs
funLL = @(theta)getLlikCRFMean(theta, ss_proteins, L, N, features_aa, seqlen_all);
theta = zeros([numel(ss_proteins), 1]);
lambdaL2 = ones(size(theta))*lambdaBar;
llTrace = NaN(options.maxIter, 1);

% Run Mean Field
[thetaML,~, ~, outputInfo] = minFunc(@penalizedL2, theta, options, funLL, lambdaL2);
llTrace(1:length(outputInfo.trace.fval), a) = outputInfo.trace.fval;

