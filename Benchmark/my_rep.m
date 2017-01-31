function a = my_rep(m)
%definitions
imgSize = 32;
imgPixel = [imgSize imgSize];
%preprocessing
rdata = im_box(m,1,0); %remove empty empty border columns and rows
rdata = im_resize(rdata, imgPixel); % resize
a = prdataset(rdata);
end