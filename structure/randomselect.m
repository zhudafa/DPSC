function W_ran = randomselect(W, K)

n = size(W,1);
[~, idx_ran] = sort(W, 2, 'descend');
W_ran = zeros(n, n);

for ii = 1:n
    W_ran(ii, idx_ran(ii, 1:K)) = W(ii, idx_ran(ii, 1:K));
end
W_ran;
