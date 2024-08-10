function W_ran = randomselect4(W,M,S,delta)
K=M*delta;
n = size(W,1);
[~, idx_ran] = sort(W, 2, 'descend');
W_ran = zeros(n, n);

for ii = 1:n
    W_ran(ii, idx_ran(ii, 1:S)) = W(ii, idx_ran(ii, 1:S));
    W_ran(ii, idx_ran(ii, randi([(M-S+1),M],1,(K-S)))) = W(ii, idx_ran(ii, randi([(M-S+1),M],1,(K-S))));
end
W_ran = (W_ran+W_ran')/2;