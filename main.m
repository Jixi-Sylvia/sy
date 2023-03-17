close all;
clc;
%% load model
addpath(genpath('pyramid transformation'));
model =  'D:/cnnmodel.mat';
load(model);
[convolution1_patchsize2,convolution1_filters] = size(weights_b1_1);  %9*64
convolution1_patchsize = sqrt(convolution1_patchsize2);
[convoluyion2_channels,convolution2_patchsize2,conv2_filters] = size(weights_b1_2);  %64*9*128
convolution2_patchsize = sqrt(convolution2_patchsize2);
[convolution3_channels,convolution3_patchsize2,convolution3_filters] = size(weights_b1_3);  %128*9*256
convolution3_patchsize = sqrt(convolution3_patchsize2);
[convolution4_channels,convolution4_patchsize2,convolution4_filters] = size(weights_output);  %512*64*2
convolution4_patchsize = sqrt(convolution4_patchsize2);
%% load images
img1= imread('D:/s01_ir.bmp');
img2= imread('D:/s01_vis.bmp');
imshow(img1);
figure;imshow(img2);
tic;

Im1 = double(img1)/255;
Im2 = double(img2)/255;
if size(Im1,3)>1
    gray1=rgb2gray(Im1);
    gray2=rgb2gray(Im2);
else
    gray1=Im1;
    gray2=Im2;
end
[height, width] = size(gray1);

%% conv1
weights_conv1 = reshape(weights_b1_1, convolution1_patchsize, convolution1_patchsize, convolution1_filters);
conv1_data1 = zeros(height, width, convolution1_filters,'single');
conv1_data2 = zeros(height, width, convolution1_filters,'single');
for i = 1 : convolution1_filters
    conv1_data1(:,:,i) = conv2(gray1, rot90(weights_conv1(:,:,i),2), 'same');
    conv1_data1(:,:,i) = max(conv1_data1(:,:,i) + biases_b1_1(i), 0);
    conv1_data2(:,:,i) = conv2(gray2, rot90(weights_conv1(:,:,i),2), 'same');
    conv1_data2(:,:,i) = max(conv1_data2(:,:,i) + biases_b1_1(i), 0);
end
%% conv2
conv2_data1 = zeros(height, width, conv2_filters,'single');
conv2_data2 = zeros(height, width, conv2_filters,'single');
for i = 1 : conv2_filters
    for j = 1 : convoluyion2_channels
        conv2_subfilter = rot90(reshape(weights_b1_2(j,:,i), convolution2_patchsize, convolution2_patchsize),2);
        conv2_data1(:,:,i) = conv2_data1(:,:,i) + conv2(conv1_data1(:,:,j), conv2_subfilter, 'same');
        conv2_data2(:,:,i) = conv2_data2(:,:,i) + conv2(conv1_data2(:,:,j), conv2_subfilter, 'same');
    end
    conv2_data1(:,:,i) = max(conv2_data1(:,:,i) + biases_b1_2(i), 0);
    conv2_data2(:,:,i) = max(conv2_data2(:,:,i) + biases_b1_2(i), 0);
end

%% max-pooling2
conv2_data1_pooling=zeros(ceil(height/2), ceil(width/2), conv2_filters,'single');
conv2_data2_pooling=zeros(ceil(height/2), ceil(width/2), conv2_filters,'single');
for i = 1 : conv2_filters    
    conv2_data1_pooling(:,:,i) = maxpooling_s2(conv2_data1(:,:,i));
    conv2_data2_pooling(:,:,i) = maxpooling_s2(conv2_data2(:,:,i));
end
%% conv3
conv3_data1 = zeros(ceil(height/2), ceil(width/2), convolution3_filters,'single');
conv3_data2 = zeros(ceil(height/2), ceil(width/2), convolution3_filters,'single');
for i = 1 : convolution3_filters
    for j = 1 : convolution3_channels
        conv3_subfilter = rot90(reshape(weights_b1_3(j,:,i), convolution3_patchsize, convolution3_patchsize),2);
        conv3_data1(:,:,i) = conv3_data1(:,:,i) + conv2(conv2_data1_pooling(:,:,j), conv3_subfilter, 'same');
        conv3_data2(:,:,i) = conv3_data2(:,:,i) + conv2(conv2_data2_pooling(:,:,j), conv3_subfilter, 'same');
    end
    conv3_data1(:,:,i) = max(conv3_data1(:,:,i) + biases_b1_3(i), 0);
    conv3_data2(:,:,i) = max(conv3_data2(:,:,i) + biases_b1_3(i), 0);
end

%% feature layer
conv3_data=cat(3,conv3_data1,conv3_data2);
conv4_data=zeros(ceil(height/2)-convolution4_patchsize+1,ceil(width/2)-convolution4_patchsize+1,convolution4_filters,'single');
for i = 1 : convolution4_filters
    for j = 1 : convolution4_channels
        conv4_subfilter = rot90((reshape(weights_output(j,:,i), convolution4_patchsize, convolution4_patchsize)),2);
        conv4_data(:,:,i) = conv4_data(:,:,i) + conv2(conv3_data(:,:,j), conv4_subfilter, 'valid');
    end
end 
%% softmax ouput layer
conv4_data=double(conv4_data);
output_data=zeros(ceil(height/2)-convolution4_patchsize+1,ceil(width/2)-convolution4_patchsize+1,convolution4_filters);
output_data(:,:,1)=exp(conv4_data(:,:,1))./(exp(conv4_data(:,:,1))+exp(conv4_data(:,:,2)));
output_data(:,:,2)=1-output_data(:,:,1);
outMap=output_data(:,:,2);

%% focus map generation
sumMap=zeros(height,width);
cntMap=zeros(height,width);
patch_size=16;
temp_size_y=patch_size;
temp_size_x=patch_size;
stride=2;
y_bound=height-patch_size+1;
x_bound=width-patch_size+1;

[h,w]=size(outMap);
for j=1:h
    jj=(j-1)*stride+1;
    if jj<=y_bound
        temp_size_y=patch_size;
    else
        temp_size_y=height-jj+1;
    end
    for i=1:w
        ii=(i-1)*stride+1;
        if ii<=x_bound
            temp_size_x=patch_size;
        else
            temp_size_x=width-ii+1;
        end
        sumMap(jj:jj+temp_size_y-1,ii:ii+temp_size_x-1)=sumMap(jj:jj+temp_size_y-1,ii:ii+temp_size_x-1)+outMap(j,i);
        cntMap(jj:jj+temp_size_y-1,ii:ii+temp_size_x-1)=cntMap(jj:jj+temp_size_y-1,ii:ii+temp_size_x-1)+1;
    end
end

focusMap=sumMap./cntMap;

%% LP
if size(Im1,3)>1
    weightMap=repmat(focusMap,[1 1 3]);
else
    weightMap=focusMap;
end

pyr = imagegaussian_pyramid(zeros(height,width));
nlev = length(pyr);

pyrW=imagegaussian_pyramid(weightMap,nlev);
pyrI1=laplaciansharpen(Im1,nlev);
pyrI2=laplaciansharpen(Im2,nlev);

for l = 1:nlev
   pyr{l}=band_fuse(pyrI1{l},pyrI2{l},pyrW{l},0.6);
end

% reconstruct
imgf = reconstruct_laplacian_pyramid(pyr);

toc;

imshow(uint8(imgf*255));
imwrite(uint8(imgf*255),'D:/fused.bmp');
