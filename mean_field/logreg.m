%logistic regression

seed = 0;
rng(seed);

%% Load data, split into train + test
directory = '../data/data_pll';
[ss_proteins, features_aa, seqlen_all, gt] = load_data(directory);
L = numel(features_aa); % seqlen variables
N = seqlen_all.*(seqlen_all - 1)/2; % Number of possible edges

%% Generate vectors
edge_vectors = cell(L, 1);

num_aa = 20*(20 + 1)/2;

for l=1:L
    vecs = zeros(N(l), num_aa + 2); %20 amino acid features, 1 dist, 1 seqlen 
    seqlen = int32(seqlen_all(l));
    vecs(:, num_aa + 2) = seqlen;
    for i = 0:(seqlen - 2)
        for j = (i+1):(seqlen - 1)
            lin_idx = i*seqlen - (i + 1)*(i+2)/2 + j + 1; %one-indexing
            vecs(lin_idx, features_aa{l}(lin_idx) + 1) = 1;
            vecs(lin_idx, num_aa + 1) = j - i; %distance feature
        end
    end
    edge_vectors{l} = vecs;
end

all_vec = double(vertcat(edge_vectors{:}));
all_gt = double(vertcat(gt{:}));
total_edges = numel(all_gt);

%% Perform logistic regression

%options.MaxIter = 1000;
%options.TolX = 1e-6;

%[b, ~, stats] = glmfit(all_vec, all_gt, 'binomial', 'link', 'logit');

%% Run Logistic Regression

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

%% Testing (on training data for now)

thetaTest = [thetaML(end); thetaML(1:end-1)];
scores = glmval(thetaTest, all_vec, 'logit');

[X, Y, T, AUC] = perfcurve(all_gt, scores, 1);
disp(AUC);
figure(1);
hold on
plot(X, Y, 'm', 'LineWidth', 2);

%% Generate image map:
t = 0.15;

directory = '../data/data_pll/test';
[ss_proteins, features_aa, seqlen, gt] = load_data(directory);
L = numel(features_aa); % seqlen variable
N = seqlen.*(seqlen - 1)/2; % Number of possible edges

edge_vectors = cell(L, 1);

num_aa = 20*(20 + 1)/2;
for l=1:L
    vecs = zeros(N(l), num_aa + 2); %20 amino acid features, 1 dist, 1 seqlen 
    seqlen = int32(seqlen);
    vecs(:, num_aa + 2) = seqlen;
    for i = 0:(seqlen - 2)
        for j = (i+1):(seqlen - 1)
            lin_idx = i*seqlen - (i + 1)*(i+2)/2 + j + 1; %one-indexing
            vecs(lin_idx, features_aa{l}(lin_idx) + 1) = 1;
            vecs(lin_idx, num_aa + 1) = j - i; %distance feature
        end
    end
    edge_vectors{l} = vecs;
end
all_vec = double(vertcat(edge_vectors{:}));
all_gt = double(vertcat(gt{:}));
total_edges = numel(all_gt);

scores = glmval(b, all_vec, 'logit');

assignments = scores > t;
adj = zeros(seqlen);
true_adj = zeros(seqlen);
edgeIndex = 1;
for i = 1:seqlen-1
    for j = i+1:seqlen
        adj(i, j) = assignments(edgeIndex);
        adj(j, i) = adj(i, j);
        true_adj(i, j) = all_gt(edgeIndex);
        true_adj(j, i) = true_adj(i, j);
        edgeIndex = edgeIndex + 1;
    end
end

adj = 1 - adj;
true_adj = 1 - true_adj;
figure(1)
imshow(adj)
figure(2)
imshow(true_adj)


