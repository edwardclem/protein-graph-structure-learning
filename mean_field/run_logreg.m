function [ output_args ] = run_logreg(train_data, test_data, outfile)

seed = 0;
rng(seed);

% Load training data
[~, features_aa, seqlen_all, gt] = load_data(train_data);
L = numel(features_aa); % seqlen variable
N = seqlen_all.*(seqlen_all - 1)/2; % Number of possible edges

% Generate vectors
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

% Perform logistic regression

%options.MaxIter = 1000;
%options.TolX = 1e-6;

b = glmfit(all_vec, all_gt, 'binomial', 'link', 'logit');

%[b, FitInfo] = lassoglm(all_vec,all_gt,'binomial','link','logit', 'lambda', 0.0);

% Testing

%load test data

[~, features_test, seqlen_test, gt_test] = load_data(test_data);
L_test = numel(features_aa);

edge_vectors_test = cell(L_test, 1);

for l=1:L_test
    vecs = zeros(N(l), num_aa + 2); %20 amino acid features, 1 dist, 1 seqlen 
    seqlen = int32(seqlen_test(l));
    vecs(:, num_aa + 2) = seqlen;
    for i = 0:(seqlen - 2)
        for j = (i+1):(seqlen - 1)
            lin_idx = i*seqlen - (i + 1)*(i+2)/2 + j + 1; %one-indexing
            vecs(lin_idx, features_test{l}(lin_idx) + 1) = 1;
            vecs(lin_idx, num_aa + 1) = j - i; %distance feature
        end
    end
    edge_vectors_test{l} = vecs;
end

all_vec_test = double(vertcat(edge_vectors_test{:}));
all_gt_test = double(vertcat(gt_test{:}));

scores = glmval(b, all_vec_test, 'logit');

[X, Y, T, AUC] = perfcurve(all_gt_test, scores, 1);
save(outfile, 'X', 'Y', 'T', 'AUC', 'b');
end

