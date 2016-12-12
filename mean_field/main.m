seed = 0;
rng(seed);

%% Load data, split into train + test
directory = 'data/test_oneprot';
[ss_proteins, features_aa, seqlen_all, gt] = load_data(directory);
L = numel(features_aa); % seqlen variable
N = seqlen_all.*(seqlen_all - 1)/2; % Number of possible edges

%% Run Mean Field CRF

% Options
lambdaBar = 0;
options.maxIter = 1000;
options.progTol = 1e-11;
crfOpt.verbose = 0; % Print things while running? 
crfOpt.nThreads = 4; % Number of threads to use
crfOpt.condDist = 0; % condition on edges with sequence distance less than this

% Setup inputs
funLL = @(theta)getLlikCRFMean(theta, ss_proteins, L, N, features_aa, seqlen_all, crfOpt);
theta = zeros([numel(ss_proteins), 1]);
lambdaL2 = ones(size(theta))*lambdaBar;
llTrace = NaN(options.maxIter, 1);

% Run Mean Field
fprintf('Starting Gradient Descent Mean Field CRF\n');
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

[X, Y, T, AUC] = perfcurve(all_gt, all_mus, 1);
% FAR = zeros(size(t_val));
% DR = zeros(size(t_val));
% for l = 1:L
%     DR_l = arrayfun(@(t) nnz((muhat{l} > t) & (gt{l} == 1))/nnz(gt{l} == 1), t_val);
%     FAR_l = arrayfun(@(t) nnz((muhat{l} > t) & (gt{l} == 0))/nnz(gt{l} == 0), t_val);
%     %figure(l)
%     %plot(FAR_l, DR_l);
%     DR = DR + DR_l;
%     FAR = FAR + FAR_l;
% end
% DR = DR/L;
% FAR = FAR/L;
figure;
hold on
% plot(FAR, DR)
plot(X, Y);
plot(0:0.1:1, 0:0.1:1);