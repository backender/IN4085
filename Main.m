%% IN4085 Pattern Recognition | Final Project

%  Instructions
%  ------------
% 
%  This file contains ...
%

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
pause;

%rdata = im_rotate(rdata, 220);
rdata = im_box(rdata,1,0) ; %remove empty empty border columns and rows 
rdata = im_resize(rdata, imgPixel); % resize
dataset = prdataset(rdata);%convert to dataset
disp([newline 'Dataset prepared and ready'])
pause;

%% Pixel Representation %%
feature_dataset = im_features(dataset, 'all');
[train_data, test_data] = gendat(feature_dataset, 0.5);

%w = ldc(f_train_data);
%acc = 1 - (f_test_data * w * testc)

w2 = ldc;
w3 = qdc;
w4 = knnc;
w5 = parzenc;
w6 = svc;
w7 = fisherc;
w8 = naivebc;

%w10 = [w2,w3,w4,w5,w6,w7,w8]*{prodc,meanc,medianc,maxc,minc,votec};


%%
W = {w2,w3,w4,w5,w6,w7,w8};%,w10};
V = train_data*W;
disp([newline 'Errors for individual classifiers'])
testc(test_data,V);

%% Feature selection
% featureReduction=5;
% Fi =train_data*featseli([],'in-in',featureReduction);
% Ff =train_data*featself([],'in-in',featureReduction);
% Fo =train_data*featselo([],'in-in',featureReduction);
% 
% Vi = train_data*Fi;
% Vf = train_data*Ff;
% Vo = train_data*Fo;
% 
% Ti = test_data*Fi;
% Tf = test_data*Ff;
% To = test_data*Fo;
% 
% disp([newline 'Errors with featseli'])
% testc(Ti,Vi);
% disp([newline 'Errors with featself'])
% testc(Tf,Vf);
% disp([newline 'Errors with featselo'])
% testc(To,Vo);

% %invidfeatures_mapping = featseli(train_data);
% %fi_train_data = train_data * invidfeatures_mapping;
% %fi_test_data = test_data * invidfeatures_mapping;
% %V2 = f_train_data*W;
% %disp([newline 'Errors with featureself for individual classifiers'])
% %testc(f_test_data,V2);
% %
% %VALL = [V2{:}];
% %         % Define combiners
% % WC = {prodc,meanc,medianc,maxc,minc,votec};
% %         % Combine (result is cell array of combined classifiers)
% % VC = VALL * WC;
% %         % Test them all
% % disp([newline 'Errors for combining rules with featureself'])
% %testc(f_test_data,VC)

%% W/ PCA 85%
p = pcam([],0.85);

w2p = p*w2;
w3p = p*qdc;
w4p = p*knnc;
w5p = p*parzenc;
w6p = p*svc;
w7p = p*fisherc;
w8p = p*naivebc;

Wp = {w2p,w3p,w4p,w5p,w6p,w7p,w8p};
Vp = train_data*Wp;
disp([newline 'Errors for individual classifiers with PCA 85'])
testc(test_data,Vp);

%% W/ PCA 99%
p = pcam([],0.99);

w2p = p*w2;
w3p = p*qdc;
w4p = p*knnc;
w5p = p*parzenc;
w6p = p*svc;
w7p = p*fisherc;
w8p = p*naivebc;

Wp = {w2p,w3p,w4p,w5p,w6p,w7p,w8p};
Vp = train_data*Wp;
disp([newline 'Errors for individual classifiers with PCA 85'])
testc(test_data,Vp);

%% Using Dissimilarity
pr = proxm([],'d',2);

w2pr = pr*w2;
w3pr = pr*qdc;
w4pr = pr*knnc;
w5pr = pr*parzenc;
w6pr = pr*svc;
w7pr = pr*fisherc;
w8pr = pr*naivebc;

Wpr = {w2pr,w3pr,w4pr,w5pr,w6pr,w7pr,w8pr};
Vpr = train_data*Wpr;
disp([newline 'Errors for individual classifiers with Dissimiarity'])
testc(test_data,Vpr);

%% Custom: Logistic Regression %%
[train_data, test_data] = gendat(dataset, 0.5);

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
ytest = str2num(ytest(:,7));
ytest = arrayfun(@(i) zeroToTen(i), ytest);


pred = predictOneVsAll(all_theta, Xtest);
%fprintf('%f', pred);
     
fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == ytest)) * 100);

