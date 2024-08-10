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
trail = 10;
m = [14,15,16,17,18,19,20];
k=[7,8,9,10,11,12];



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i = 1:11
    corruption = (i-1)*0.1;
    fprintf("\n===Synthetic-corruption(%.1f): ===\n", corruption);
  for kk=1:length(k)
      k_kk = k(kk);
  for mm=1:length(m)
    m_mm = m(mm);
delta = k_kk/m_mm;  
    for j = 1:trail
        %fprintf("Trial %d: ", j);
         [data, Y] = dataGenerator_subspaceData(500, 50, 5);
        X = addGaussianNoise1(data',corruption ,0)';
        
        %%% Data processing
        X = NormalizeFea(X, 1);    %%% Normalization
        data_num = size(X, 1);
        class_num = length(unique(Y));
        
        %%% Affinity learning
        CMat = admmOutlier_mat_func(X', affine, alpha);
        C = CMat(1:data_num,:);
        W_SSC = BuildAdjacency(thrC(C,1));
        W_SDN = DiffusionNewchangeeta(W_SSC, m_mm,class_num,k_kk, delta, class_num); 
 
     
        %%% Spectral clustering
  
        Y_SDN = SpectralClustering(W_SDN, class_num);
        
        
        %%% Check accuracy
        
        acc_SDN(j) = clusteringAcc(Y_SDN, Y);
        
        nmi_SDN(j) = nmi(Y_SDN, Y);
       
    end
    
    fprintf("=======================================================================================\n")
    fprintf("K(%.1f),M(%.1f), Corruption(%.1f) ACC-average: SDN(%.3f) \n",...
        k(kk),m(mm), corruption, mean(acc_SDN));
    fprintf("K(%.1f),M(%.1f), Corruption(%.1f) STD: SDN(%.3f)\n",...
       k(kk),m(mm), corruption, std(acc_SDN));
    fprintf("=======================================================================================\n")
    fprintf("K(%.1f),M(%.1f), Corruption(%.1f) NMI-average: SDN(%.3f)\n",...
       k(kk), m(mm), corruption, mean(nmi_SDN));
    fprintf("K(%.1f),M(%.1f), Corruption(%.1f) STD: SDN(%.3f)\n",...
       k(kk), m(mm), corruption, std(nmi_SDN));
    fprintf("=======================================================================================\n")
    
end
  end
end
