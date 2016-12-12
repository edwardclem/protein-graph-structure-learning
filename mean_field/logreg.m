%logistic regression

seed = 0;
rng(seed);

%% Load data, split into train + test
directory = 'data/data_parallel';
[ss_proteins, features_aa, seqlen_all, gt] = load_data(directory);
L = numel(features_aa); % seqlen variable
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

b = glmfit(all_vec, all_gt, 'binomial', 'link', 'logit');

%[b, FitInfo] = lassoglm(all_vec,all_gt,'binomial','link','logit', 'lambda', 0.0);

%% Testing (on training data for now)


scores = glmval(b, all_vec, 'logit');

[X, Y, T, AUC] = perfcurve(all_gt, scores, 1);
disp(AUC);
figure(1);
plot(X, Y);


