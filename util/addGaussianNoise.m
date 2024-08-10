function x=addGaussianNoise(x,p,p1)

% add p percentage noise to x%

[m,n]=size(x);
N=ceil(n*p);
idx=ceil(rand(1,N)*n);
if (length(idx)~=0)
    for i=1:length(idx)
        x(:,idx(i))=x(:,idx(i))+normrnd(0,0.3*norm(x(:,idx(i))),m,1);
    end
end
corruption_mask = randperm( m*n, round( p1*m*n ) );
a=zeros(size(x));
a(corruption_mask)=x(corruption_mask);
%a = imnoise(a,'salt & pepper',0.01);%Ω∑—Œ‘Î“Ù
%a = imnoise(a,'speckle',0.01);%≥À–‘‘Î“Ù
a = imnoise(a,'gaussian');%∏ﬂÀπ‘Î“Ù
x(corruption_mask)=a(corruption_mask);
%X = imnoise(X,'poisson');%≤¥À…‘Î“Ù
return