function A = DiffusionNewchangeeta(W,M,L,K,delta,class_num)

n = size(W, 1);
d = sum(W, 2);
D = diag(d + eps);
eta=2;
% Pre-processing of weight matrix W
W = W - diag(diag(W)) + D;   %%% use node degree as self-affinity
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% Normalization  %%%%%%%%%%%%%%%%%
% S = W ./ repmat(sum(W, 2)+eps, 1, n);

d = sum(W,2);
D = diag(d + eps);
W = D^(-1/2)*W*D^(-1/2);      %%% Symmetric normalization is better
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

S = knnSparse(W, M);%%% Sparse is important for affinity matrix
WW = randomselect(S,K);
P = randomselect4(W,M,L,delta);
P2 = randomselect4(W,M,L,delta);
maxIter_out = 20;
maxIter_in = 50;
epsilon = 5e-2;
%epsilon = 5e-4;        %%% convergence threshold
%%% Initialization
Z = zeros(n,n);
A = WW;
%alpha = 0.99;
for t = 1:maxIter_out
     
    %%% Update A
    A_old = A;
for ii = 1:(maxIter_in/2)
    temp = (A + eta*Z)*P;
    eta=eta*0.5;
    %temp = alpha*P*(A + Z)*P' + (1-alpha)*eye(length(WW));
    %temp = WW*P;
    if norm(temp-A,'fro') < epsilon, break; end  
    A = temp;   
end
%P2 = randomselect4(WW, M, K);
for ii = (maxIter_in/2+1):maxIter_in
    %temp = alpha*P2*(A + Z)*P2' + (1-alpha)*eye(length(WW));
    temp = (A + eta*Z)*P2;
    %temp =  WW*P2 ;
    eta=eta*0.5;
    if norm(temp-A,'fro') < epsilon, break; end  
    A = temp;   
end
    err = norm(A - A_old, 'fro');
     %fprintf(" err(%.2f)...\n", err);
    if err < epsilon, break; end
    
    % update Z
    Z_old=Z;
    Z = label_similarity(A, class_num); 
    d = sum(Z,2);
    D = diag(d + eps);
    Z = D^(-1/2)*Z*D^(-1/2);
    Z = knnSparse(Z, K); 
    %a = 10;
    Z_new=Z;
    err1 = norm(Z_new - Z_old, 'fro');
    % fprintf(" err(%.2f)...err1(%.2f)...\n",  err,err1);
end


%% Post-processing, useful for spectral clustering
A = A - diag(diag(A));
A = (A + A')/2;
