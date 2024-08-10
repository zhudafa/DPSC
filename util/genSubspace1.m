function [X, s] = genSubspace1(D, n, Ni, di, sigma, corruption, corruption1)
% This code generates a matrix X of D by sum(Ni) that contains n subspaces
% of dimension di, with noise level sigma.
% Input:
%     D: dimension of ambience space
%     n: number of subspace
%     Ni: #points in each subspace
%     di: dimension of each subspace
%     sigma: noise deviation
% Output:
%     X: data (D by sum(Ni))
%     s: label of X (1 by sum(Ni))
% example: 
%     [X, s] = genSubspace(100, 6, 100, 5, 0);

% Copyright Chong You @ Johns Hopkins University, 2016
% chong.you1987@gmail.com

if length(Ni) == 1
    Ni = repmat(Ni, 1, n);
end
if length(di) == 1
    di = repmat(di, 1, n);
end
if ~exist('sigma', 'var')
    sigma = 0;
end
if ~exist('corruption', 'var')
    corruption = 0;
end

if ~exist('corruption1', 'var')
    corruption1 = 0;
end
X = zeros(D, sum(Ni)); s = zeros(1, sum(Ni));
idx = 0;
for in = 1:n
%     Xtmp = randn(D, di(in)) * randn(di(in), Ni(in));

    Xtmp = randn(D, D);
    [Utmp, ~, ~] = svds(Xtmp, di(in)); % generate a random subspace ???
    Vtmp = randn(di(in), Ni(in));
    Xtmp = Utmp * Vtmp; % generate random points in subspace
    Xtmp = bsxfun(@rdivide, Xtmp, sqrt(sum(Xtmp .^2, 1)) ); % normalize
    
    X(:, [idx+1 : idx+Ni(in)]) = Xtmp;
    s([idx+1 : idx+Ni(in)]) = in;
    idx = idx+Ni(in);
end

noise_term = sigma * randn(D, sum(Ni)) / sqrt(D);
X = X + noise_term;
%X = imnoise(X,'salt & pepper',corruption);%Ω∑—Œ‘Î“Ù
corruption_mask = randperm( D*sum(Ni), round( corruption*D*sum(Ni) ) );
a=zeros(size(X));
a(corruption_mask)=X(corruption_mask);
a = imnoise(a,'salt & pepper',0.01);%Ω∑—Œ‘Î“Ù
%a = imnoise(a,'speckle',0.01);%≥À–‘‘Î“Ù
X(corruption_mask)=a(corruption_mask);
%X = imnoise(X,'poisson');%≤¥À…‘Î“Ù
%corruption_mask = randperm( D*sum(Ni), round( corruption*D*sum(Ni) ) );
%X(corruption_mask) = 0;
if corruption1 ~= 0
    a = randperm(size(X,2));
    b = a(1:size(X,2)*corruption1);
    
    for i = 1:length(b)
        j = b(i);
        X(:,j) = X(:,j) + sqrt(0.1*norm(X(:,j)))*randn(1, 1);
        %size(sqrt(0.1*norm(X(:,j))))
    end
   
end
end