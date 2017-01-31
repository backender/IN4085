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

rdata = prnist([0:9],[1:50:1000]);
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
w8 = svc(proxm('p',5));
w9 = svc(proxm('h'));
w10 = svc(proxm('e'));
w11 = svc(proxm('s'));

%w10 = [w2,w3,w4,w5,w6,w7,w8]*{prodc,meanc,medianc,maxc,minc,votec};

%% LIVE TEST

W = {w2,w3,w4,w5,w6,w7};
V = train_data*W;

setTotal = prdataset();
setTotal.name = 'LIVE Total';

for d = 0:5 % digits

    set = prdataset();
    set.name = 'LIVE';

    for i = 0:9 % number of samples per digit

        % Read images from folder
        I = imread(sprintf('/Users/marc/Documents/MATLAB/PRProject/digits/%d_%d.png', d, i));

        % 1. Rgb image as grayscale
        % 2. Binarize the image such representation consists of [0, 1]
        % 3. Complement as prnist training set was complemented as well
        Ibw = imcomplement(imbinarize(rgb2gray(I)));
        %figure
        %imshowpair(I,Ibw,'montage')

        Img = Ibw;

        % Remove zeros from rows and columns (alternative to im_box, which doesn't work with matrices)
        Img( ~any(Img,2), : ) = [];  %rows
        Img( :, ~any(Img,1) ) = [];  %columns

        % Resize to same format as training data
        Img = im_resize(Img, imgPixel);

        %figure
        %show(Img)

        ImgVec = im2obj(Img);
        set = [set; ImgVec];

    end

    %labels = repmat({sprintf('digit_%d', d)},1,i); % i times digit d
    ImgSet = setlabels(set,sprintf('digit_%d', d));
    setTotal = [setTotal; ImgSet];

disp([newline sprintf('Live test testc for digit %d', d)])
testc(ImgSet, V); % test against nist trained

end

disp([newline sprintf('Total Live test testc for digits 0-%d', d)])
testc(setTotal, V); % test against nist trained

disp([newline sprintf('Total Live test crossval for digit 0-%d', d)])
prcrossval(setTotal,W); % crossval with custom-handwritten only


pause;

%%
W = {w2,w3,w4,w5,w6,w7};
%V = train_data*W;
disp([newline 'Errors for individual classifiers'])
prcrossval(train_data,W);

%%
disp([newline 'Errors for combined parzen&svc classifiers']);
prcrossval(train_data,([w5,w6]*{prodc,meanc,medianc,maxc,minc,votec}));

%%
disp([newline 'Errors for combined knn&svc classifiers']);
Vcomb2 = train_data*([w4,w6]*{prodc,meanc,medianc,maxc,minc,votec});
testc(test_data,Vcomb2);

%% Feature selection
% featureReduction=500;
% Fi =train_data*featseli([],'NN',featureReduction);
% Vi = train_data * Fi * W;
% Ti = test_data*Fi;
% disp([newline 'Errors with featseli'])
% testc(Ti,Vi);
% 
% featureReduction=200;
% Fi =train_data*featseli([],'NN',featureReduction);
% Vi = train_data * Fi * W;
% Ti = test_data*Fi;
% disp([newline 'Errors with featseli'])
% testc(Ti,Vi);
% 
% featureReduction=100;
% Fi =train_data*featseli([],'NN',featureReduction);
% Vi = train_data * Fi * W;
% Ti = test_data*Fi;
% disp([newline 'Errors with featseli'])
% testc(Ti,Vi);
% 
% featureReduction=50;
% Fi =train_data*featseli([],'NN',featureReduction);
% Vi = train_data * Fi * W;
% Ti = test_data*Fi;
% disp([newline 'Errors with featseli'])
% testc(Ti,Vi);
% 
% featureReduction=25;
% Fi =train_data*featseli([],'NN',featureReduction);
% Vi = train_data * Fi * W;
% Ti = test_data*Fi;
% disp([newline 'Errors with featseli'])
% testc(Ti,Vi);

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

%% W/ PCA 25%
p = pcam([],0.25);
Wp = p*W;
disp([newline 'Errors for individual classifiers with PCA 25'])
prcrossval(train_data,Wp);

%% W/ PCA 85%
p = pcam([],0.85);
Wp = p*W;
disp([newline 'Errors for individual classifiers with PCA 85'])
prcrossval(train_data,Wp);

%% W/ PCA 50%
p = pcam([],0.5);
Wp = p*W;
disp([newline 'Errors for individual classifiers with PCA 50'])
prcrossval(train_data,Wp);

%% W/ PCA 99%
p = pcam([],0.99);
Wp = p*W;
disp([newline 'Errors for individual classifiers with PCA 99'])
prcrossval(train_data,Wp);

%% Using Dissimilarity
%pr = proxm([],'d',2);
pr = distm;
Wpr = pr*W;
disp([newline 'Errors for individual classifiers with Dissimiarity'])
prcrossval(train_data,Wp);

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

