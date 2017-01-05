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

rdata = prnist([0:9],[1:50:1000]);
%rdata = im_rotate(rdata, 220);
rdata = im_box(rdata,1,0) ; %remove empty empty border columns and rows 
rdata = im_resize(rdata, imgPixel); % resize
dataset = prdataset(rdata);%convert to dataset

[train_data, test_data] = gendat(dataset, 0.5);

X = getdata(train_data);
y = getlab(train_data);
y = str2num(y(:,7));
y = arrayfun(@(i) zeroToTen(i), y);


for i=1:length(y)
    %X(i,:) = reshape(imfill(reshape(X(i,:), imgPixel)), [1 input_layer_size]);
end


%% ============ Part 2: Feature Extraction ============
%  ...


%% ============ Part 3: Display example of dataset ============
% Randomly select 100 data points to display
m = size(X, 1);
rand_indices = randperm(m);
sel = X(rand_indices(1:100), :);

displayData(sel);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ============ Part 4: Logistic Regression ============
%  ...

fprintf('\nTraining One-vs-All Logistic Regression...\n')

lambda = 0.1;
[all_theta] = oneVsAll(X, y, num_labels, lambda);


%% ================ Part 5: Predict for One-Vs-All ================

Xtest = getdata(test_data);
ytest = getlab(test_data);
ytest = str2num(ytest(:,7));
ytest = arrayfun(@(i) zeroToTen(i), ytest);

%for i=1:length(ytest)
%    Xtest(i,:) = reshape(imfill(reshape(Xtest(i,:), imgPixel)), [1 input_layer_size]);
%end

pred = predictOneVsAll(all_theta, Xtest);
%fprintf('%f', pred);
     
fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == ytest)) * 100);

