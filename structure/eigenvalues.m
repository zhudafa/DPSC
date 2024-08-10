clc;
clear;

addpath('util/');
addpath('L1_ADMM/');


%%% Hyperparameters to be set
affine = 0;
alpha = 10;
m = 20;
k = 10;
trail = 10;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i = [2,7]
    corruption = (i-1)*0.1;
    fprintf("\n===Synthetic-corruption(%.1f): ===\n", corruption);
    
        [data, Y] = dataGenerator_subspaceData(500, 50, 5);
        X = addGaussianNoise(data', corruption)';
        
        %%% Data processing
        X = NormalizeFea(X, 1);    %%% Normalization
        data_num = size(X, 1);
        class_num = length(unique(Y));
        
        CMat = admmOutlier_mat_func(X', affine, alpha);
        C = CMat(1:data_num,:);
        W_SSC = BuildAdjacency(thrC(C,1));
        W_SDN = DiffusionNeweig(W_SSC, m, k);
        D = Spectralvalue(W_SDN);
        d = diag (D);
        E(:,i) = sort (d);
        a = [1: length(E(:,1))];
        l = E(5,i) - E(4,i)
end

sz=25;
c = linspace(1,10,length(E(:,2)));
subplot(1,2,1) %几行几列第几个图
scatter(a,E(:,2),'filled')
xlim([0 20])
ylim([0 1])
ylabel('eigenvalue')
legend("corruption(10)") %为图像加上标题
subplot(1,2,2)
scatter(a,E(:,7),'filled')
xlim([0 20])
ylim([0 1])
legend("corruption(60)") %为图像加上标题
xlabel('x')
ylabel('eigenvalue') %在y坐标上增加批注