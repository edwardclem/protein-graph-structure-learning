seed = 0;
rng(seed);

%% Load data, split into train + test
directory = '../data/data_parallel';
[ss_proteins, features_aa, seqlen_all, gt] = load_data_SGD(directory);
L = numel(features_aa); % seqlen variable
N = seqlen_all.*(seqlen_all - 1)/2; % Number of possible edges

%% Run Mean Field CRF

% Options
lambdaBar = 0;
options.maxIter = 1000;
options.progTol = 1e-11;
crfOpt.verbose = 0; % Print things while running? 
crfOpt.nThreads = 4; % Number of threads to use
crfOpt.condDist = 0;

% Run Mean Field
fprintf('Starting Stochastic Gradient Descent Mean Field CRF\n');
tstart = tic;
[thetaML] = minimizeLL_SGD(ss_proteins, L, N, features_aa, seqlen_all, crfOpt);
tstop = toc(tstart);
fprintf('Gradient Descent Elapsed in %0.1fs.\n', tstop);
llTrace(1:length(outputInfo.trace.fval)) = outputInfo.trace.fval;

%% Plot results
muhat = margProbMean(thetaML, N, features_aa, seqlen_all, crfOpt); % change to test data
t_val = 1:-0.001:0.001;
FAR = zeros(size(t_val));
DR = zeros(size(t_val));
for l = 1:L
    DR = DR + arrayfun(@(t) nnz((muhat{l} > t) & (gt{l} == 1))/nnz(gt{l} == 1), t_val);
    FAR = FAR + arrayfun(@(t) nnz((muhat{l} > t) & (gt{l} == 0))/nnz(gt{l} == 0), t_val);
end
DR = DR/L;
FAR = FAR/L;
figure;
hold on
plot(FAR, DR)
plot(0:0.1:1, 0:0.1:1);