%% IN4085 Pattern Recognition | Final Project

%  Instructions
%  ------------
% 
%  This file contains ...
%  
%   Tests for scenario 2

%% Initialization
clear ; close all; clc

%% Setup the parameters
imgSize = 32;

input_layer_size  = imgSize^2;  % 20x20 Input Images of Digits
num_labels = 10;          % 10 labels, from 1 to 10   
                          % (note that we have mapped "0" to label 10)
imgPixel = [imgSize imgSize];

%% =========== Part 1: Data Selection & Preprocessing =============

rdata = prnist([0:9],[1:5:1000]);
disp([newline 'Data ready'])
%pause;

%rdata = im_rotate(rdata, 220);
rdata = im_box(rdata,1,0); %remove empty empty border columns and rows
rdata = im_resize(rdata, imgPixel); % resize

dataset = prdataset(rdata);%convert to dataset
disp([newline 'Dataset prepared and ready'])
%pause;

%% Pixel Representation %%
%feature_dataset = im_features(dataset, 'all');
[train_data, test_data] = gendat(dataset, 0.5);

%w = ldc(f_train_data);
%acc = 1 - (f_test_data * w * testc)

w2 = ldc;
w3 = knnc([], 3);
w4 = parzenc;
w5 = svc(proxm('p',5));
w6 = svc(proxm('h',5));
w7 = bpxnc;
%w10 = [w2,w3,w4,w5,w6,w7,w8]*{prodc,meanc,medianc,maxc,minc,votec};


%%
W = {w2, w3, w4, w5, w6, w7};%,w10};
V = train_data*W;
disp([newline 'Errors for individual classifiers'])
testc(test_data,V);

 %% Feature selection
 featureReduction=800;
 Fi = train_data*featseli([],'NN',featureReduction);
 Vi = train_data * Fi * {w3};
 Ti = test_data*Fi;
 disp([newline 'Errors with featseli 800'])
 testc(Ti,Vi);
 
 featureReduction=800;
 Fi = train_data*featseli([],'maha-s',featureReduction);
 Vi = train_data * Fi * {w2,w4,w5,w6,w7};
 Ti = test_data*Fi;
 disp([newline 'Errors with featseli'])
 testc(Ti,Vi);
 
  featureReduction=600;
 Fi = train_data*featseli([],'NN',featureReduction);
 Vi = train_data * Fi * {w3};
 Ti = test_data*Fi;
 disp([newline 'Errors with featseli 600'])
 testc(Ti,Vi);
 
 featureReduction=600;
 Fi = train_data*featseli([],'maha-s',featureReduction);
 Vi = train_data * Fi * {w2,w4,w5,w6,w7};
 Ti = test_data*Fi;
 disp([newline 'Errors with featseli'])
 testc(Ti,Vi);

%% W/ PCA 75%
 p = pcam([],0.45);
 Wp = p*{w2, w3, w4, w5, w6, w7};
 Vp = train_data*Wp;
 disp([newline 'Errors for individual classifiers with PCA 55'])
 testc(test_data,Vp);
%% W/ PCA 75%
 p = pcam([],0.55);
 Wp = p*{w2, w3, w4, w5, w6, w7};
 Vp = train_data*Wp;
 disp([newline 'Errors for individual classifiers with PCA 55'])
 testc(test_data,Vp);
 %% W/ PCA 75%
 p = pcam([],0.65);
 Wp = p*{w2, w3, w4, w5, w6, w7};
 Vp = train_data*Wp;
 disp([newline 'Errors for individual classifiers with PCA 65'])
 testc(test_data,Vp);
%% W/ PCA 75%
 p = pcam([],0.75);
 Wp = p*{w2, w3, w4, w5, w6, w7};
 Vp = train_data*Wp;
 disp([newline 'Errors for individual classifiers with PCA 75'])
 testc(test_data,Vp);
 
%% W/ PCA 85%
p = pcam([],0.85);
Wp = p*{w2, w3, w4, w5, w6, w7};
Vp = train_data*Wp;
disp([newline 'Errors for individual classifiers with PCA 85'])
testc(test_data,Vp);

%% W/ PCA 95%
 p = pcam([],0.95);
 Wp = p*{w2, w3, w4, w5, w6, w7};
 Vp = train_data*Wp;
 disp([newline 'Errors for individual classifiers with PCA 95'])
 testc(test_data,Vp);

%% Using Dissimilarity
% %pr = proxm([],'d',2);
% pr = distm;
% Wpr = pr*{w2, w3, w4, w5, w6, w7};
% Vpr = train_data*Wpr;
% disp([newline 'Errors for individual classifiers with Dissimiarity'])
% testc(test_data,Vpr);