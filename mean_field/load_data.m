function [ss_proteins, features_aa, seqlen_all, gt] = load_data(directory)
nAA = 20;
fileset = dir([directory '*.mat']);
features_aa = cell(numel(fileset), 1);
seqlen_all = zeros([numel(fileset), 1]);
ss_proteins = zeros([1, 4 + nAA*(nAA+1)/2 + 3]);
gt = cell(numel(fileset), 1);
for idx = 1:numel(fileset)
    load([directory fileset(idx).name], 'aa_index', 'edge_density', 'true_edges', 'sum_true_dist', 'seqlen')
    features_aa{idx} = uint32(aa_index); % determine what eddie calls these
    ss_proteins(1:4) = ss_proteins(1:4) + double(edge_density);
    ss_proteins(5:end-3) = ss_proteins(5:end-3) + histcounts(aa_index, 0:nAA*(nAA+1)/2);
    ss_proteins(end-2) = ss_proteins(end-2) + sum_true_dist;
    nEdges = sum(true_edges);
    ss_proteins(end-1) = ss_proteins(end-1) + nEdges*seqlen;
    ss_proteins(end) = ss_proteins(end) + nEdges;
    seqlen_all(idx) = seqlen;
    gt{idx} = true_edges;
end
seqlen_all = uint32(seqlen_all);
ss_proteins = ss_proteins';

end