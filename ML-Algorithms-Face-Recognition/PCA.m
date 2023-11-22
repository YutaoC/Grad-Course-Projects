function [Test_D_P,Train_D_P,n] = PCA(Train_Data,Test_Data,lamda)
%
% This function implement the Principle Component Analysis.
%
% [Test_D_P,Train_D_P] = PCA(Train_Data,Test_Data,lamda)
%
% Train_Data - The training data (Must be reshaped to a vector)
% Test_Data - The testing data (Must be reshaped to a vector)
% Lamda - A designed parameter to decide the number of dimensions.
%         Lager lamda meas more dimensions (The values should be within (0,1))
% Test_D_P - The projected training data (each column is an observation)
% Train_D_P - The projected testing data (each column is an observation)
%
% Yutao Chen
% 15/11/2018
%
    %Get the size of the data
    [N_Pixel,N_Train] = size(Train_Data);
    [~,N_Test] = size(Test_Data);
    
    %Center the training data
    data_centered = zeros(N_Pixel,N_Train);
    for i = 1:N_Pixel
        data_centered(i,:) = Train_Data(i,:) - mean(Train_Data(i,:));
    end
    
    %Calculate the sigma matrix
    sigma = zeros(N_Pixel,N_Pixel);
    for j = 1:N_Train
        sigma = sigma + data_centered(:,j)*(data_centered(:,j))';
    end
    sigma = sigma / (N_Train);
    
    %Calculate the eigenvalues and eigenvectors
    [eigvec,eigval] = eig(sigma);
    eigval_vec = eigval*ones(N_Pixel,1); %reshape it into a vector
    [eigval_vec_s,index] = sort(eigval_vec,'descend'); %sort in decreasing order
    
    %Decide how many dimensions should take
    total = sum(eigval_vec_s);%total sum of all the eigenvalues
    total_tmp = 0;
    for n = 1:N_Pixel
        total_tmp = total_tmp + eigval_vec_s(n);
        if total_tmp >= lamda * total %when the sum reached the threshold
            break %stop the iteration
        end
    end
    u = zeros(N_Pixel,n);
    for m = 1:n
        u(:,m) = eigvec(:,index(m));
    end
    
    %Project the training and testing data respectively
    Train_D_P = zeros(n,N_Train);
    Test_D_P = zeros(n,N_Test);
    for j = 1:N_Train
        Train_D_P(:,j) = u.' * Train_Data(:,j);
    end
    for i = 1:N_Test
        Test_D_P(:,i) = u.' * Test_Data(:,i); 
    end
end
