% Brain tumor segmentation using image thresholding (Solidity concept)
%

clc;
clear;
close;
img=imread('HGG(high-grade glioma) brain.jpg');
bw=im2bw(img,0.9);
label=bwlabel(bw);
stats=regionprops(label,'solidity','Area');
density=[stats.Solidity];  %Solidity concept
area=[stats.Area];   %Area calculation
high_dense_area=density>0.5;    %Taking high dense area of white pixels
max_area=max(area(high_dense_area));
tumor_label=find(area==max_area);
tumor=ismember(label,tumor_label);
se=strel('square',7);     %Giving boundary to tumor as a square shape
tumor=imdilate(tumor,se);
figure(1),subplot(1,3,1);
imshow(img,[]);
title('brain');
subplot(1,3,2);
imshow(tumor,[]);
title('tumor alone');
[B,L]=bwboundaries(tumor,'noholes');
subplot(1,3,3);
imshow(img,[]);
title('tumor alone');
hold on;
for i=1:length(B)
    plot(B{1}(:,2),B{1}(:,1),'Y','linewidth',1.45)  %x axis and y axis of the square, yellow color, line width=1.45
end
title('detected tumor');hold off;


%% Lung tumor detection

a=imread('lung.jpg');
figure,
imshow(a);
title('Input image');
try
    Dimg=rgb2gray(imread('lung.jpg'));
catch
    Dimg=imread('lung.jpg');
end
imdata=reshape(Dimg,[],1);
imdata=double(imdata);
[IDX,nn]=kmeans(imdata,3);
imIDX=reshape(IDX,size(Dimg));
figure(2),
imshow(imIDX,[]);
title('Index image');
figure(3),
subplot(3,2,1),imshow(imIDX==1,[]);
subplot(3,2,2),imshow(imIDX==2,[]);
subplot(3,2,3),imshow(imIDX==3,[]);
subplot(3,2,4),imshow(imIDX==4,[]);
subplot(3,2,5),imshow(imIDX==5,[]);
%%
bw=(imIDX==1);  %from figure 3 choose the imIDX value where tumor is present 
se=ones(5);
bw=imopen(bw,se);
bw=bwareaopen(bw,400);
figure(4),imshow(bw);
title('Segmented tumor');


binary=imbinarize(a,0.555);
figure(5),
imshow(binary);
title('Binary image');
% Get rid of particles smaller than 1000 pixels.
f = bwareaopen(binary, 900);
figure(6),
imshow(f);
title('getting rid of smaller pixel values');