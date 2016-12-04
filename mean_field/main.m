seed = 0;

rng(seed);
addpath('./hw5data')
addpath(genpath('./minFunc_2012'))

%% Load data, split into train + test
directory = '../data/summed_suffstats/';
[ss_proteins, gt] = load_data(directory);
data_train = ss_proteins(1:6,:);
data_test = ss_proteins(7:end,:);
L = size(ss_proteins, 2);
N = ss_proteins(end-4,:); % seqlen variable
N = N.*(N - 1)/2; % Number of possible edges
[ss, feats] = suffStatsCRF(data_train);

%% Run Mean Field CRF

% Options
lambdaBar = 0.01;
options.maxIter=1000;

% Setup inputs
funLL = @(theta)getLlikCRFMean(theta, ss, L, N, feats);
theta = zeros([numel(ss), 1]);
lambdaL2 = ones(size(theta))*lambdaBar;
llTrace = NaN(options.maxIter, numel(algs));

% Run Mean Field
[thetaML,~, ~, outputInfo] = minFunc(@penalizedL2, theta, options, funLL, lambdaL2);
llTrace(1:length(outputInfo.trace.fval), a) = outputInfo.trace.fval;

%% Question 2f
% TODO: Complete margProbMean.m, getLlikMRFMean.m, getLlikCRFMean.m

tic;
dataset = 1; % run on dataset 1
algsInd = 6; 
[algs2,aucAll2,llTrace2] = runexp(dataset,algsInd);

% combine results to display
algs=[algs,algs2];
llTrace=cat(2,llTrace,llTrace2);
aucAll=cat(3,aucAll,aucAll2);

makePlots(algs,aucAll,llTrace,dataset,2);
t2 = toc;

%% Question 2g
tic;
dataset=2; % run on dataset 2
algsInd = [1,4,5,6]; 
[algs3,aucAll3,llTrace3] = runexp(dataset,algsInd);
makePlots(algs3,aucAll3,llTrace3,dataset,3);
t3=toc;
