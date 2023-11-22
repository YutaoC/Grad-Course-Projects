function accuracy = Bayes(Train_D_P,Test_D_P,L_Test)
%
% This function implement the Bayes' classifier using the Maximum Likelihood.
%
% accuracy = Bayes(Train_D_P,Test_D_P,L_Test)
%
% Train_D_P - training data (each column is an observation)
% Test_D_P - testing data (each column is an observation)
% L_Test - true label of testing data
% accuracy - accuracy of the classifier (the value is within (0,1))
%
% Yutao Chen
% 15/11/2018
%
    %Get the size of the data and initialize the parameters
    [N_Pixel,N_Data_P] = size(Train_D_P);
    [~,N_Test] = size(Test_D_P);
    N_C = length(unique(L_Test)); % number of classes
    M = zeros(N_Pixel,N_C); % mean value matrix
    C = zeros(N_Pixel,N_Pixel,N_C); %covariance matrix
    C_inv = zeros(N_Pixel,N_Pixel,N_C); %inverse covariance matrix
    N_D_C = N_Data_P/N_C; % # of data per class
    
    %Calculate the mean and covariance of the training data
    for i = 1:N_C
        start = 1+N_D_C*(i-1);
        stop = N_D_C*i;
        M(:,i) = mean(Train_D_P(:,start:stop),2);
        C(:,:,i) = cov(transpose(Train_D_P(:,1+N_D_C*(i-1):N_D_C*i)));
        C(:,:,i) = C(:,:,i) + 1 * eye(N_Pixel); % covariance matrix singularity
        C_inv(:,:,i) = inv(C(:,:,i));
    end
    
    %Calculate the probabilities of each class for every testing data
    P_0 = zeros(N_C,1);
    P_1 = zeros(N_Pixel,N_C);
    for j = 1:N_C %Terms that only depend on the class
        P_0(j) = -0.5 * M(:,j)'*C_inv(:,:,j)*M(:,j)-0.5*log(det(C(:,:,j)));
        P_1(:,j) = C_inv(:,:,j) * M(:,j);
    end
    res = zeros(N_Test,1);
    for i = 1:N_Test
        max_prob = -inf(1);
        for j = 1:N_C
            tmp =  -0.5 * Test_D_P(:,i)' * C_inv(:,:,j) * Test_D_P(:,i);
            prob = tmp + P_0(j) + P_1(:,j)'*Test_D_P(:,i);
            if prob > max_prob %take the maximum probability
                max_prob = prob;
                res(i) = j;
            end
        end
    end
    
    %Calculate the accuracy
    correct = 0;
    for i = 1:N_Test
        if res(i) == L_Test(i)
            correct = correct + 1;
        end
    end
    accuracy = correct / N_Test;
end