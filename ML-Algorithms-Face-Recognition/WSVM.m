function [H,x,y,alpha,bias] = WSVM(Train_Data,L_Train,Wn,mu)
%
% This function implement a weak SVM based on part of the training data
%
% [H,x,y,alpha,bias] = WSVM(Train_Data,L_Train,Wn,mu)
%
% Train_Data - all the training data
% L_Train - correspongding true label
% Wn - correspongding weights
% mu - a number between 0 and 1 decides the percentage of data, larger mu
%      means taking large percentage of data
% H - the value before take the sign function
% x - support vectors
% y - the corresponding true label of support vectors
% alpha - the weights of different support vectors in the final classifier
% bias - bias term in final classifier
%
% Yutao Chen
% 16/11/2018
%
    %Get the size of data and initializations
    [N_Pixel,N_Train] = size(Train_Data);
    [Wn_sort,index] = sort(Wn,'descend'); %sort the weight in decreasing order
    J = zeros(N_Train,1);
    C = 0.01;
    tmp = 0;
    
    %Choose part of the training data
    for i = 1:N_Train
       tmp = tmp + Wn_sort(i); 
       if tmp <= mu %take account the data when the sum is smaller than mu (the sum of total weights are 1)
          J(i) = index(i);
       else
           cnt = i-1;
           break
       end
       cnt = i;
    end
    Train_D_N = zeros(N_Pixel,cnt);
    L_Train_N = zeros(1,cnt);
    Wn_N = zeros(1,cnt);
    for j = 1:cnt
        Train_D_N(:,j) = Train_Data(:,J(j)); %take the part of data
        L_Train_N(:,j) = L_Train(:,J(j)); %corresponding label
        Wn_N(j) = Wn(J(j)); %corresponding weights
    end
    
    %Calculate the liner SVM
    [H,~,x,y,alpha,bias] = Kernel_SVM(Train_D_N,Train_Data,L_Train_N,L_Train,C);
end