function [ss_proteins, gt] = load_data(directory)
nAA = 20;
fileset = dir([directory '*.mat']);
features_aa = cell(numel(fileset), 1);
ss_proteins = zeros([4 + nAA*(nAA+1)/2 + 3, 1]);
gt = cell(numel(fileset), 1);
for idx = 1:numel(fileset)
    load([directory fileset(idx).name], 'feats', 'edge_density', 'true_edges', 'totDist', 'nEdges', 'seqlen')
    features_aa{idx} = feats; % determine what eddie calls these
    ss_proteins(1:4) = ss_proteins(1:4) + edge_density;
    ss_proteins(5:end-3) = ss_proteins(5:end-3) + histcounts(feats, 1:nAA*(nAA+1)/2);
    ss_proteins(end-2) = ss_proteins(end-2) + totDist;
    ss_proteins(end-1) = ss_proteins(end-1) + nEdges*seqlen;
    ss_proteins(end) = ss_proteins(end) + nEdges;
    gt{idx} = true_edges;
end

end