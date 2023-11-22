# face-recognition
implemented a simple face recognition using some classic classifiers

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
### main.m   
This file includes all the different ways to form training data anf testing data    
It also contain all calls of functions     

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   
### Bayes.m   
accuracy = Bayes(Train_D_P,Test_D_P,L_Test)   

This function implement the Bayes' classifier using the Maximum Likelihood   

Train_D_P - training data (each column is an observation)  
Test_D_P - testing data (each column is an observation)  
L_Test - true label of testing data  
accuracy - accuracy of the classifier (the value is within (0,1))   

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   
### K_NN.m   
accuracy = K_NN(Train_D_P,Test_D_P,L_Test,K)   

This function implement the k-NN rule   

Train_D_P - training data (each column is an observation)  
Test_D_P - testing data (each column is an observation)  
L_Test - true label of testing data  
K - a designed parameter which decides the number of neighborhood  
accuracy - accuracy of the classifier (the value is within (0,1))  

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   
### LDA.m  
\[Test_D_P,Train_D_P\] = LDA(Train_Data,Test_Data,N_C)  

This function implement the Linear Discriminant Analysis  

Train_Data - The training data (Must be reshaped to a vector)  
Test_Data - The testing data (Must be reshaped to a vector)  
N_C - Number of classes in the dataset  
Test_D_P - The projected training data (each column is an observation)  
Train_D_P - The projected testing data (each column is an observation)  

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   
### PCA.m  
\[Test_D_P,Train_D_P\] = PCA(Train_Data,Test_Data,lamda)  

This function implement the Principle Component Analysis   

Train_Data - The training data (Must be reshaped to a vector)  
Test_Data - The testing data (Must be reshaped to a vector)  
Lamda - A designed parameter to decide the number of dimensions  
             Lager lamda meas more dimensions (The values should be within (0,1))  
Test_D_P - The projected training data (each column is an observation)  
Train_D_P - The projected testing data (each column is an observation)  

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
### Kernel_SVM.m  
\[res_val,accuracy,x,y,alpha,bias\] = Kernel_SVM(Train_D_P,Test_D_P,L_Train,L_Test,C)  

This function implement kernel SVM with two different kernels(RBF and Polynomial).The change of kernel and parameters must be done in the function   

Train_D_P - training data  
Test_D_P - testing data  
L_Train - true label of training data  
L_Test - true label of testing data  
C - regualrization parameter control the misclassification of each training sample. Larger C means a smaller-margin hyperplane  
res_val - the value before take the sign function  
accuracy - accuracy of the classifier  
x - support vectors  
y - the corresponding true label of support vectors  
alpha - the weights of different support vectors in the final classifier  
bias - bias term in final classifier  

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   
### kernel.m  
 K = kernel(x1,x2,func_type,kernal_params)   

This function implement three different kernels which are  
1. Linear Kernel  
2. Radial Basis Function (RBF) Kernel  
3. Polynominal Kernel   

x1 - data set one  
x2 - data set two  
func_type - 'l' Linear Kernel  
            'g' Radial Basis Function (RBF) Kernel   
            'p' Polynominal Kernel   
kernel_params - kernel parameter  

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  
### Boosted_SVM.m  
\[accuracy,r,alpha_opt,res,Wn_N\] = Boosted_SVM(Train_D_P,Test_D_P,L_Train,L_Test,mu)  

This function implement boosted SVM  

Train_D_P - training data   
Test_D_P - testing data  
L_Train - true label of training data  
L_Test - true label of testing data  
mu - the parameter used in function WSVM  
T - number of weak SVM classifers  
accuracy - accurace of the classifier (a number between 0 and 1)   
r - perdicted label  
alpha_opt - optimal of wieghts of different weak SVM classifier  
res - the value before take the sign function  

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   
### WSVM.m   
\[H,x,y,alpha,bias\] = WSVM(Train_Data,L_Train,Wn,mu)  

This function implement a weak SVM based on part of the training data   

Train_Data - all the training data  
L_Train - correspongding true label  
Wn - correspongding weights  
mu - a number between 0 and 1 decides the percentage of data, larger mu means taking large percentage of data   
H - the value before take the sign function  
x - support vectors  
y - the corresponding true label of support vectors  
alpha - the weights of different support vectors in the final classifier  
bias - bias term in final classifier   

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   
### Test_WSVM.m  
res = Test_WSVM(x,y,alpha,bias,Test_Data,L_Test)  

This function test the weak SVM on the training data   

x - support vectors   
y - the corresponding true label of support vectors  
alpha - the weights of different support vectors in the final classifier  
bias - bias term in final classifier  
Test_Data - the data to test the weak SVM  
L_Test - the correponding true label of the data  
res - the resulting value before take the sign function  

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   
