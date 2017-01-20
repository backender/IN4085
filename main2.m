%scenario 2

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
disp([newline 'Data ready'])
%pause;

%rdata = im_rotate(rdata, 220);
rdata = im_box(rdata,1,0); %remove empty empty border columns and rows 
rdata = im_resize(rdata, imgPixel); % resize
dataset = prdataset(rdata);%convert to dataset
disp([newline 'Dataset prepared and ready'])
%pause;

[train_data, test_data] = gendat(dataset, 0.5);

%% W/ PCA 85%
p = pcam([],0.85);
Wp = p*svc(proxm('p',5));
Vp = train_data*Wp;
disp([newline 'Errors for individual classifiers with PCA 85'])
testc(test_data,Vp);

%% LIVE TEST

% Read images from folder
I = imread('C:\Users\Ron\git\IN4085\digits\1_0.png');

% 1. Rgb image as grayscale
% 2. Binarize the image such representation consists of [0, 1]
% 3. Complement as prnist training set was complemented as well
Ibw = imcomplement(imbinarize(rgb2gray(I)));
%figure
%imshowpair(I,Ibw,'montage')git

% Resize to same format as training data
Img = im_resize(Ibw, imgPixel);

%ImgVec = im2obj(Img);
ImgVec = Img(:)';
% Predict ...?
labels = ['digit_0'];
data = [ImgVec];
test_data = prdataset(data, labels);
Vp = train_data*Wp;
disp([newline 'Prediction for live test with PCA 85'])
testc(test_data, Vp)
test_data*Vp