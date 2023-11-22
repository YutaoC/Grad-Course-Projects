function res = Test_WSVM(x,y,alpha,bias,Test_Data,L_Test)
%
% This function test the weak SVM on the training data
%
% res = Test_WSVM(x,y,alpha,bias,Test_Data,L_Test)
%
% x - support vectors
% y - the corresponding true label of support vectors
% alpha - the weights of different support vectors in the final classifier
% bias - bias term in final classifier
% Test_Data - the data to test the weak SVM
% L_Test - the correponding true label of the data
% res - the resulting value before take the sign function
%
% Yutao Chen
% 16/11/2018
%
    res = zeros(1,length(L_Test));
    K_Test = kernel(x,Test_Data,'l',100);
    for h = 1:length(L_Test)
        res(h) = sum(alpha.*y.*K_Test(h,:))+bias;
    end
end