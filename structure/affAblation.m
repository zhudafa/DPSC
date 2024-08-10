clc;
clear;
%addpath('toolbox');
addpath('util/');
addpath('L1_ADMM/');
addpath('structure\');
addpath('Diffusion\');
%%% Hyperparameters to be set
affine = 0;
alpha = 10;
m = 12;
k=10;
delta = 10/12;
trail = 1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i = 3
    corruption = (i-1)*0.1;
    fprintf("\n===Synthetic-corruption(%.1f): ===\n", corruption);
    
    for j = 1:trail
        %fprintf("Trial %d: ", j);
        [data, Y] = dataGenerator_subspaceData(800, 50, 8);
        X = addGaussianNoise1(data',corruption ,0)';
        
        %%% Data processing
        X = NormalizeFea(X, 1);    %%% Normalization
        data_num = size(X, 1);
        class_num = length(unique(Y));
        
        %%% Affinity learning
        CMat = admmOutlier_mat_func(X', affine, alpha);
        C = CMat(1:data_num,:);
        W_SSC = BuildAdjacency(thrC(C,1));
        W_SSCTPG = IterativeDiffusionTPGKNN(W_SSC, k);
        W_SDN1 = DiffusionNewchange(W_SSC, m,class_num,k, delta);
        W_SDN = DiffusionNewchange2(W_SSC, m,class_num,k, delta, class_num); 
 figure, imagesc(W_SSC);
figure, imagesc(W_SSCTPG);
figure, imagesc(W_SDN1);
figure, imagesc(W_SDN);     
        %%% Spectral clustering
        Y_SSC = SpectralClustering(W_SSC, class_num);
        Y_SSCTPG = SpectralClustering(W_SSCTPG, class_num);
        Y_SDN1 = SpectralClustering(W_SDN1, class_num);
        Y_SDN = SpectralClustering(W_SDN, class_num);
        
        
        %%% Check accuracy
        acc_SSC(j) = clusteringAcc(Y_SSC, Y);
        acc_SSCTPG(j) = clusteringAcc(Y_SSCTPG, Y);
        acc_SDN1(j) = clusteringAcc(Y_SDN1, Y);
        acc_SDN(j) = clusteringAcc(Y_SDN, Y);
        
     
        nmi_SSC(j) = nmi(Y_SSC, Y);
        nmi_SSCTPG(j) = nmi(Y_SSCTPG, Y);
        nmi_SDN1(j) = nmi(Y_SDN1, Y);
        nmi_SDN(j) = nmi(Y_SDN, Y);
        
      
       
      
        fprintf("ACC-SSC(%.3f), SSCTPG(%.3f), SDN1(%.3f)£¬SDN(%.3f)\n",...
            acc_SSC(j), acc_SSCTPG(j), acc_SDN1(j), acc_SDN(j));

    end
    
    
end
