%% data.mat
N_C = 200;
N_Train = 200*2;
N_Test = 200*1;
N_Pixel = 24*21;
Train_Data = zeros(N_Pixel,N_Train);
Test_Data = zeros(N_Pixel,N_Test);
L_Train = zeros(N_Train,1);
L_Test = zeros(N_Test,1);
count1 = 1;
count2 = 1;
for i = 1:N_C
   for j = 0:2:2
      Train_Data(:,count1) = reshape(face(:,:,3*i-j),[N_Pixel,1]); %choose different training data
      L_Train(count1) = i; %record the label
      count1 = count1 +1;
   end
   for k = 3:3
      Test_Data(:,count2) = reshape(face(:,:,3*i-1),[N_Pixel,1]); %take the remianing as testing data
      L_Test(count2) = i; %record the label
      count2 = count2 + 1;
   end
end

% [Test_D_P,Train_D_P,n] = PCA(Train_Data,Test_Data,0.9); %PCA
% [Test_D_P,Train_D_P] = LDA(Train_Data,Test_Data,N_C); %LDA
% accuracy = Bayes(Train_D_P,Test_D_P,L_Test); %BAYES
% accuracy = K_NN(Train_D_P,Test_D_P,L_Test,1); %KNN
display(accuracy)

%% illumination.mat
N_C = 68;
N_Train = 68*19;
N_Test = 68*2;
N_Pixel = 1920;
Train_Data = zeros(N_Pixel,N_Train);
Test_Data = zeros(N_Pixel,N_Test);
L_Train = zeros(N_Train,1);
L_Test = zeros(N_Test,1);
count1 = 1;
count2 = 1;
for i = 1:N_C
   for j = 1:19
      Train_Data(:,count1) = illum(:,j,i);%choose different training data
      L_Train(count1) = i; %record the label
      count1 = count1 +1;
   end
   for k = 20:21
      Test_Data(:,count2) = illum(:,k,i);%take the remianing as testing data
      L_Test(count2) = i; %record the label
      count2 = count2 + 1;
   end
end

% [Test_D_P,Train_D_P] = PCA(Train_Data,Test_Data,0.9); %PCA
% [Test_D_P,Train_D_P] = LDA(Train_Data,Test_Data,N_C); %LDA
% accuracy = Bayes(Train_D_P,Test_D_P,L_Test); % BAYES
% accuracy = K_NN(Train_D_P,Test_D_P,L_Test,4); %KNN
display(accuracy)

%% pose.mat
N_C = 68;
N_Train = 68*12;
N_Test = 68*1;
N_Pixel = 48*40;
Train_Data = zeros(N_Pixel,N_Train);
Test_Data = zeros(N_Pixel,N_Test);
L_Train = zeros(N_Train,1);
L_Test = zeros(N_Test,1);
count1 = 1;
count2 = 1;
for i = 1:N_C
   for j = 1:12
      Train_Data(:,count1) = reshape(pose(:,:,j,i),[N_Pixel,1]);%choose different training data
      L_Train(count1) = i;%record the label
      count1 = count1 +1;
   end
   for k = 13:13
      Test_Data(:,count2) = reshape(pose(:,:,k,i),[N_Pixel,1]);%take the remianing as testing data
      L_Test(count2) = i;%record the label
      count2 = count2 + 1;
   end
end

% [Test_D_P,Train_D_P] = PCA(Train_Data,Test_Data,0.90); %PCA
% [Test_D_P,Train_D_P] = LDA(Train_Data,Test_Data,N_C); %LDA
% accuracy = Bayes(Train_D_P,Test_D_P,L_Test); %BAYES
% accuracy = K_NN(Train_D_P,Test_D_P,L_Test,4); %KNN
display(accuracy)

%% data.mat kernal_SVM
N_Train = 2*150;
N_Test = 2*50;
N_Pixel = 24*21;
Train_Data = zeros(N_Pixel,N_Train);
Test_Data = zeros(N_Pixel,N_Test);
L_Train = zeros(1,N_Train);
L_Test = zeros(1,N_Test);
count1 = 1;
count2 = 1;
for i = 1:150
   for j = 1:2
      Train_Data(:,count1) = reshape(face(:,:,3*(i-1)+j),[N_Pixel,1]); %first 150 Neural and Expression
      if j == 1
          L_Train(count1) = 1; %label of Neutal is 1 
      else
          L_Train(count1) = -1;%label of Expression is -1 
      end
      count1 = count1 +1;
   end
end
for i = 151:200 
   for j = 1:2
      Test_Data(:,count2) = reshape(face(:,:,3*(i-1)+j),[N_Pixel,1]);%remaining Neural and Expression
      if j == 1
          L_Test(count2) = 1;%label of Neutal is 1 
      else
          L_Test(count2) = -1;%label of Expression is -1 
      end
      count2 = count2 +1;
   end
end

[Test_D_P,Train_D_P] = PCA(Train_Data,Test_Data,0.90); %PCA
[~,accuracy,~,~,~,~] = Kernel_SVM(Train_D_P,Test_D_P,L_Train,L_Test,0.5); %Kernel SVM
display(accuracy)

%% data.mat Boosted_SVM
N_Train = 2*150;
N_Test = 2*50;
N_Pixel = 24*21;
Train_Data = zeros(N_Pixel,N_Train);
Test_Data = zeros(N_Pixel,N_Test);
L_Train = zeros(1,N_Train);
L_Test = zeros(1,N_Test);
count1 = 1;
count2 = 1;
for i = 1:150
   for j = 1:2
      Train_Data(:,count1) = reshape(face(:,:,3*(i-1)+j),[N_Pixel,1]);%first 150 Neural and Expression
      if j == 1
          L_Train(count1) = 1;%label of Neutal is 1 
      else
          L_Train(count1) = -1;%label of Expression is -1 
      end
      count1 = count1 +1;
   end
end
for i = 151:200 
   for j = 1:2
      Test_Data(:,count2) = reshape(face(:,:,3*(i-1)+j),[N_Pixel,1]);%remaining Neural and Epression
      if j == 1
          L_Test(count2) = 1;%label of Neutal is 1 
      else
          L_Test(count2) = -1;%label of Expression is -1 
      end
      count2 = count2 +1;
   end
end

[Test_Data,Train_Data] = PCA(Train_Data,Test_Data,0.9); %PCA
[accuracy,r,alpha_opt,res] = Boosted_SVM(Train_Data,Test_Data,L_Train,L_Test,0.6,25); %Boosted SVM
display(accuracy)
%% data.mat
N_C = 2;
N_Train = 2*150;
N_Test = 2*50;
N_Pixel = 24*21;
Train_Data = zeros(N_Pixel,N_Train);
Test_Data = zeros(N_Pixel,N_Test);
L_Train = zeros(1,N_Train);
L_Test = zeros(1,N_Test);
count1 = 1;
count2 = 1;
for i = 1:150
      Train_Data(:,count1) = reshape(face(:,:,3*(i-1)+1),[N_Pixel,1]);%first 150 Neural
      L_Train(count1) = 1; %label of Neutal is 1 
      count1 = count1 +1;
end
for i = 1:150
      Train_Data(:,count1) = reshape(face(:,:,3*(i-1)+2),[N_Pixel,1]);%first 150 Expression
      L_Train(count1) = 2;%label of Expression is 2
      count1 = count1 +1;
end
for i = 151:200 
      Test_Data(:,count2) = reshape(face(:,:,3*(i-1)+1),[N_Pixel,1]); %remaining Neural
      L_Test(count2) = 1;%label of Neutal is 1 
      count2 = count2 +1;
end
for i = 151:200 
      Test_Data(:,count2) = reshape(face(:,:,3*(i-1)+2),[N_Pixel,1]); %remaining Epression
      L_Test(count2) = 2;%label of Expression is 2
      count2 = count2 +1;
end

% [Test_D_P,Train_D_P] = PCA(Train_Data,Test_Data,0.9); %PCA
% [Test_D_P,Train_D_P] = LDA(Train_Data,Test_Data,N_C); %LDA
% accuracy = Bayes(Train_D_P,Test_D_P,L_Test); %BAYES
% accuracy = K_NN(Train_D_P,Test_D_P,L_Test,17); %KNN
display(accuracy)