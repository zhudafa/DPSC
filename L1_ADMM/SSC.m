%--------------------------------------------------------------------------
% This is the function to call the sparse optimization program, to call the
% spectral clustering algorithm and to compute the clustering error.
% r = projection dimension, if r = 0, then no projection
% affine = use the affine constraint if true
% s = clustering ground-truth
% missrate = clustering error
% CMat = coefficient matrix obtained by SSC
%--------------------------------------------------------------------------
% Copyright @ Ehsan Elhamifar, 2012
%--------------------------------------------------------------------------

function [Mis1,Mis2, Mis3] = SSC(X,r,affine,alpha,outlier,rho,s)
%[Mis1,Mis2,Mis3] = SSC(X,0,0,20,1,1,s)
if (nargin < 6)
    rho = 1;
end
if (nargin < 5)
    outlier = false;
end
if (nargin < 4)
    alpha = 20;
end
if (nargin < 3)
    affine = false;
end
if (nargin < 2)
    r = 0;
end

n = max(s);
Xp = DataProjection(X,r);

if (~outlier)
    CMat = admmLasso_mat_func(Xp,affine,alpha);
    C = CMat;
else
    CMat = admmOutlier_mat_func(Xp,affine,alpha);
    N = size(Xp,2);
    C = CMat(1:N,:);
end
C1 = abs(C) + abs(C)';
CKSym = BuildAdjacency(thrC(C,rho));
M = IterativeDiffusionTPGKNN(C1,10);
A = ADP(C1, 10, n);

grps1 = SpectralClustering(CKSym,n);
grps2 = SpectralClustering(M, n);
grps3 = SpectralClustering(A, n);

Mis1 = clusteringAcc(grps1, s);
Mis2 = clusteringAcc(grps2, s);
Mis3 = clusteringAcc(grps3, s);
return
