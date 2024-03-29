seed = 0;
rng(seed);

%% Load data, split into train + test
directory = '../data/data_pll';
[ss_proteins, features_aa, seqlen_all, gt] = load_data(directory);
L = numel(features_aa); % seqlen variable
N = seqlen_all.*(seqlen_all - 1)/2; % Number of possible edges

% Debug
%ss_proteins(3:4) = [total2 total3]';

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
funLL = @(theta)getLlikCRFPll(theta, gt, L, features_aa, seqlen_all, crfOpt);
theta = zeros([numel(ss_proteins), 1]);
lambdaL2 = ones(size(theta))*lambdaBar;
llTrace = NaN(options.maxIter, 1);

% Run Mean Field
fprintf('Starting Gradient Descent Pseudo log-likelihood CRF\n');
tstart = tic;
[thetaML,~, ~, outputInfo] = minFunc(@penalizedL2, theta, options, funLL, lambdaL2);

tstop = toc(tstart);
fprintf('Gradient Descent Elapsed in %0.1fs.\n', tstop);
llTrace(1:length(outputInfo.trace.fval)) = outputInfo.trace.fval;

%% Plot results
muhat = margProbMean(thetaML, N, features_aa, seqlen_all, crfOpt); % change to test data
%scores = glmval(b, all_vec, 'logit');
all_mus = vertcat(muhat{:});
all_gt = vertcat(gt{:});

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

%% Generate image map:
directory = '../data/data_pll/test';
[~, feats_test, seqlen_test, gt_test] = load_data(directory);
gt_test = gt_test{1};
L = numel(feats_test); % seqlen variable
N = seqlen_test.*(seqlen_test - 1)/2; % Number of possible edges

muhat = margProbMean(thetaML, N, feats_test, seqlen_test, crfOpt);
scores = muhat{1};

%%
t = mean(scores) + 1.5*std(scores);
assignments = scores > t;
adj = zeros(seqlen_test);
true_adj = zeros(seqlen_test);
edgeIndex = 1;
for i = 1:seqlen_test-1
    for j = i+1:seqlen_test
        adj(i, j) = assignments(edgeIndex);
        adj(j, i) = adj(i, j);
        true_adj(i, j) = gt_test(edgeIndex);
        true_adj(j, i) = true_adj(i, j);
        edgeIndex = edgeIndex + 1;
    end
end

adj = 1 - adj;
true_adj = 1 - true_adj;
figure(1)
imshow(adj)
axis on
set(gca,'XAxisLocation','top')
figure(2)
imshow(true_adj)
set(gca,'XAxisLocation','top')
axis on
