%% IN4085 Pattern Recognition | Final Project

%  Instructions
%  ------------
% 
%  This file contains ...
%  Parameters:
%   - pixels
%   - classifier parameters
%   - pca
%   - feature selection
%   - dissimilarity

%% Initialization
clear ; close all; clc

%% Setup the parameters
imgSize = 32;

input_layer_size  = imgSize^2;  % 20x20 Input Images of Digits
num_labels = 10;          % 10 labels, from 1 to 10   
                          % (note that we have mapped "0" to label 10)
imgPixel = [imgSize imgSize];

%% =========== Part 1: Data Selection & Preprocessing =============

rdata = prnist([0:9],[1:25:1000]);
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
w3 = qdc;
w4 = knnc([],3);
w5 = parzenc([],0.25);
w6 = svc;
w7 = fisherc;
w8 = naivebc;

%w10 = [w2,w3,w4,w5,w6,w7,w8]*{prodc,meanc,medianc,maxc,minc,votec};


%%
W = {w2,w3,w4,w5,w6,w7,w8};%,w10};
V = train_data*W;
disp([newline 'Errors for individual classifiers'])
testc(test_data,V);

%%
disp([newline 'Errors for combined parzen&svc classifiers']);
Vcomb = train_data*([w5,w6]*{prodc,meanc,medianc,maxc,minc,votec});
testc(test_data,Vcomb);

%%
disp([newline 'Errors for combined knn&svc classifiers']);
Vcomb2 = train_data*([w4,w6]*{prodc,meanc,medianc,maxc,minc,votec});
testc(test_data,Vcomb2);

%% Feature selection
% featureReduction=50;
% Fi =train_data*featseli([],'NN',featureReduction);
% Vi = train_data * Fi * W;
% Ti = test_data*Fi;
% disp([newline 'Errors with featseli'])
% testc(Ti,Vi);
% 
% Ff =train_data*featself([],'NN',featureReduction);
% Vf = train_data * Ff * W;
% Tf = test_data*Ff;
% disp([newline 'Errors with featself'])
% testc(Tf,Vf);
% 
% Fo =train_data*featselo([],'NN',featureReduction);
% Vo = train_data * Fo * W;
% To = test_data*Fo;
% disp([newline 'Errors with featselo'])
% testc(To,Vo);

%% W/ PCA 85%
p = pcam([],0.85);
Wp = p*{w2,w3,w4,w5,w6,w7,w8};
Vp = train_data*Wp;
disp([newline 'Errors for individual classifiers with PCA 85'])
testc(test_data,Vp);

%% W/ PCA 99%
% p = pcam([],0.99);
% Wp = p*{w2,w3,w4,w5,w6,w7,w8};
% Vp = train_data*Wp;
% disp([newline 'Errors for individual classifiers with PCA 99'])
% testc(test_data,Vp);

%% Using Dissimilarity
%pr = proxm([],'d',2);
pr = distm;
Wpr = pr*{w2,w3,w4,w5,w6,w7,w8};
Vpr = train_data*Wpr;
disp([newline 'Errors for individual classifiers with Dissimiarity'])
testc(test_data,Vpr);

%% Custom: Logistic Regression dataset preprocessing %%

X = getdata(train_data);
y = getlab(train_data);
y = str2num(y(:,7));
y = arrayfun(@(i) zeroToTen(i), y);


%% ============ Part 3: Display example of dataset ============
% Randomly select 100 data points to display
m = size(X, 1);
%rand_indices = randperm(m);
%sel = X(rand_indices(1:100), :);

%displayData(sel);

%fprintf('Program paused. Press enter to continue.\n');
%pause;

%% ============ Part 4: Train Logistic Regression ============
%  ...

fprintf('\nTraining One-vs-All Logistic Regression...\n')

lambda = 0.1;
[all_theta] = oneVsAll(X, y, num_labels, lambda);


%% ================ Part 5: Predict/Test Logistic Regression with One-Vs-All ================

Xtest = getdata(test_data);
ytest = getlab(test_data);
% In order to validate predictions we confront the labels 
% with the prediction. Therefore the format needs to be the same. 
% That is, ?digit_X"(:,7) would return ?X?. Where X is a number 
% between 0-9
ytest = str2num(ytest(:,7));
ytest = arrayfun(@(i) zeroToTen(i), ytest);


pred = predictOneVsAll(all_theta, Xtest);
%fprintf('%f', pred);
     
fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == ytest)) * 100);

%% LIVE TEST

% Read images from folder
I = imread('/Users/marc/Documents/MATLAB/PRProject/digits/0_0.png');

% 1. Rgb image as grayscale
% 2. Binarize the image such representation consists of [0, 1]
% 3. Complement as prnist training set was complemented as well
Ibw = imcomplement(imbinarize(rgb2gray(I)));
%figure
%imshowpair(I,Ibw,'montage')

% Resize to same format as training data
Img = im_resize(Ibw, imgPixel);

%ImgVec = im2obj(Img);
ImgVec = Img(:)';

% Predict ...?