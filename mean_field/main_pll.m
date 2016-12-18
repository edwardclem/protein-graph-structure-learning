seed = 0;
rng(seed);

%% Load data, split into train + test
directory = '../data/data_test';
[ss_proteins, features_aa, seqlen_all, gt] = load_data(directory);
L = numel(features_aa); % seqlen variable
N = seqlen_all.*(seqlen_all - 1)/2; % Number of possible edges

% Debug
ss_proteins(3:4) = [total2 total3]';

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
fprintf('Starting Gradient Descent Pseudo log-likelihood CRF\n');
tstart = tic;
[thetaML,~, ~, outputInfo] = minFunc(@penalizedL2, theta, options, funLL, lambdaL2);

% maxIter = 100; % Number of passes through the data set
% stepSize = 1e-6;
% theta = zeros([size(ss_proteins, 1), 1]);
% old_val = inf;
% for iter = 1:maxIter
%     [f,g] = getLlikCRFPll(theta, gt, ss_proteins, ...
%         L, N, features_aa, seqlen_all, ...
%         crfOpt);
%     
%     fprintf('\tIter = %d of %d (fsub = %f)\n',iter,maxIter,f);
%     if (any(isinf(g)) || f > old_val)
%         stepSize = stepSize/2;
%     else
%         theta = theta - stepSize*g;
%         old_val = f;
%     end
% end
tstop = toc(tstart);
fprintf('Gradient Descent Elapsed in %0.1fs.\n', tstop);
llTrace(1:length(outputInfo.trace.fval)) = outputInfo.trace.fval;

%% Plot results
muhat = margProbMean(thetaTest, N, features_aa, seqlen_all, crfOpt); % change to test data


t_val = 1:-0.001:0.001;

all_mus = vertcat(muhat{1:end});
all_gt = vertcat(gt{1:end});

[X, Y, T, AUC] = perfcurve(all_gt,  all_mus, 1);
disp(AUC)
%load logit_and_pll.mat
load logitROC.mat
figure(1);
hold on
plot(X, Y, 'b', 'LineWidth', 2);
plot(Xlogit, Ylogit, 'm', 'LineWidth', 2);
plot(0:0.1:1, 0:0.1:1, 'r--', 'LineWidth', 2);
legend('Pll', 'Logistic', 'Random', 'Location', 'SouthEast')