function [accuracy,r,alpha_opt,res] = Boosted_SVM(Train_D_P,Test_D_P,L_Train,L_Test,mu,T)
%
% This function implement boosted SVM
%
% [accuracy,r,alpha_opt,res,Wn_N] = Boosted_SVM(Train_D_P,Test_D_P,L_Train,L_Test,mu)
%
% Train_D_P - training data
% Test_D_P - testing data
% L_Train - true label of training data
% L_Test - true label of testing data
% mu - the parameter used in function WSVM
% T - number of weak SVM classifers
% accuracy - accurace of the classifier (a number between 0 and 1)
% r - perdicted label
% alpha_opt - optimal of wieghts of different weak SVM classifier
% res - the value before take the sign function
%
% Yutao Chen
% 16/11/2018
%
    %Get the size of data and initialization
    [~,N_Train] = size(Train_D_P);
    [~,N_Test] = size(Test_D_P);
    Wn = ones(1,N_Train)/N_Train;
    res = zeros(T,length(L_Test));
    alpha_opt = zeros(1,T);
    alpha_opt(1) = 1;
    Wn_N = zeros(T,N_Train);
    
    %First weak SVM
    [H,x,y,alpha,bias] = WSVM(Train_D_P,L_Train,Wn,mu);
    res(1,:) = Test_WSVM(x,y,alpha,bias,Test_D_P,L_Test);
    h = H;
    Wn_N(1,:) = Wn;
    
    %Adaboost algorithm
    for t = 2:T
        Weighted_E = 0;
        for i = 1:length(h)
           Weighted_E = Weighted_E + Wn_N(t-1,i)*abs(sign(h(i))+L_Train(i))/2; % calculate the weighted erroe
        end 
        for i = 1:length(h)
           Wn_N(t,i) =  Wn_N(t-1,i)*((1-Weighted_E)/Weighted_E)^(-0.5*L_Train(i)*sign(h(i))); %update the weights of different data
        end
        z = sum(Wn_N(t,:));
        Wn_N(t,:) = Wn_N(t,:)./ z; % to make the weights as a distribution
        
        %Use the new wights to do weak SVM again 
        [h,x,y,alpha,bias] = WSVM(Train_D_P,L_Train,Wn_N(t,:),mu);
        res(t,:) = Test_WSVM(x,y,alpha,bias,Test_D_P,L_Test); % test the weak SVM
        
        %Find optimal alpha
        min_w_e = inf(1);
        for alpha_tmp = 0:0.05:1
            H_tmp = h.*alpha_tmp+H.*(1-alpha_tmp); %updata the result
            for i = 1:N_Train
                Weighted_E = Weighted_E + Wn_N(t,i)*abs(sign(H_tmp(i))+L_Train(i))/2; %calculate the weighted error
            end
            if Weighted_E < min_w_e %take the alpha with minimum weighted error
                min_w_e = Weighted_E;
                alpha_opt(t) = alpha_tmp;
            end
            Weighted_E = 0; %reset the weighted error
        end
        
        H = alpha_opt(t)*h+(1-alpha_opt(t))*H; %update the result
    end
    r = zeros(size(res));
    for i = 1:t
       r(i,:) = alpha_opt(i)*res(i,:); %combine the result of different weak SVMs with different weights
       l_p = sign(sum(r));
    end
    
    %Calculate the accuracy
    correct = 0;
    for i = 1:N_Test
        if l_p(i) == L_Test(i)
            correct = correct + 1;
        end
    end
    accuracy = correct / N_Test;
end