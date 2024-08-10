function Z = label_similarity(CKSym, n)

[~, D] = SpectralClustering(CKSym,n);

Z = 1 ./ (1 + D);
Z = Z ./ repmat(sum(Z, 2)+eps, 1, size(Z, 2));
Z = Z * Z';

