function [Test_D_P,Train_D_P] = LDA(Train_Data,Test_Data,N_C)
%
% This function implement the Linear Discriminant Analysis.
%
% [Test_D_P,Train_D_P] = LDA(Train_Data,Test_Data,N_C)
%
% Train_Data - The training data (Must be reshaped to a vector)
% Test_Data - The testing data (Must be reshaped to a vector)
% N_C - Number of classes in the dataset.
% Test_D_P - The projected training data (each column is an observation)
% Train_D_P - The projected testing data (each column is an observation)
%
% Yutao Chen
% 15/11/2018
%
    %Get the size of the data and initialize the parameters
    [N_Pixel,N_Train] = size(Train_Data);
    [~,N_Test] = size(Test_Data);
    N_D_C = N_Train/N_C; %# of samples per class
    
    %Calculate the mean value
    mu = zeros(N_Pixel,N_C); % mean value of each class
    mu_all = zeros(N_Pixel,1); %overall mean value
    for j = 1:N_C % mean value of each class
        for i = 1+N_D_C*(j-1):N_D_C*j
            mu(:,j) = mu(:,j) + Train_Data(:,i); 
        end
        mu(:,j) = mu(:,j)/N_D_C;
    end
    for i = 1:N_Train %overall mean value
        mu_all = mu_all + Train_Data(:,i);
    end
    mu_all = mu_all/N_Train;
    
    %Calculate between-scatter matrix
    sigma_b = zeros(N_Pixel,N_Pixel);
    for j = 1:N_C
        tmp = mu(:,j) - mu_all;
        sigma_b = sigma_b + (tmp * tmp.');
    end
    
    %Calculate within-scatter matrix
    sigma_m = zeros(N_Pixel,N_Pixel);
    sigma_w = zeros(N_Pixel,N_Pixel);
    for j = 1:N_C
        start = 1+N_D_C*(j-1);
        for k = 1:N_D_C
            diff = Train_Data(:,start+k-1) - mu(:,j);
            sigma_m = sigma_m + diff*diff.';
        end
        sigma_m = sigma_m + 1 * eye(N_Pixel);%matrix singularity
        sigma_w = sigma_w + sigma_m;
    end
    
    %Take the first N_C-1 eigenvectors
    [eigvec,~] = eigs(sigma_b,sigma_w,N_C-1);

    %Project the training and testing data respectively
    Train_D_P = zeros(N_C-1,N_Train);
    Test_D_P = zeros(N_C-1,N_Test);
    for i = 1:N_Train
        Train_D_P(:,i) = eigvec.' * Train_Data(:,i);
    end
    for i = 1:N_Test
        Test_D_P(:,i) = eigvec.' * Test_Data(:,i); 
    end
end