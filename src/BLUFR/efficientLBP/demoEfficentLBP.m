%% Set demo parameters: tets image and filter parameters
imgRGB=imread('F:\cse802_data\lfw_mtcnn_cropped\LFW_Aaron-Eckhart\LFW_Aaron_Eckhart_0001.jpg');
% imgRGB=imread('F:\cse802_data\lfw_mtcnn_cropped\LFW_Aaron-Guiel\LFW_Aaron_Guiel_0001.jpg');

img=rgb2gray(imgRGB);
nFiltSize=8;
nFiltRadius=1;
filtR=generateRadialFilterLBP(nFiltSize, nFiltRadius);

fprintf('Here is our filter:\n')
disp(filtR);

%% Test regular LBP vs RI-LBP
effLBP= efficientLBP(img, 'filtR', filtR, 'isRotInv', false, 'isChanWiseRot', false);
effRILBP= efficientLBP(img, 'filtR', filtR, 'isRotInv', true, 'isChanWiseRot', false);

uniqueRotInvLBP=findUniqValsRILBP(nFiltSize);
tightValsRILBP=1:length(uniqueRotInvLBP);
% Use this function with caution- it is relevant only if 'isChanWiseRot' is false, or the
% input image is single-color/grayscale
effTightRILBP=tightHistImg(effRILBP, 'inMap', uniqueRotInvLBP, 'outMap', tightValsRILBP);

binsRange=(1:2^nFiltSize)-1;
figure;
subplot(2,1,1)
hist(single( effLBP(:) ), binsRange);
axis tight;
title('Regular LBP hsitogram', 'fontSize', 16);

subplot(2,2,3)
hist(single( effRILBP(:) ), binsRange);
axis tight;
title('RI-LBP sparse hsitogram', 'fontSize', 16);

subplot(2,2,4)
hist(single( effTightRILBP(:) ), tightValsRILBP);
axis tight;
title('RI-LBP tight hsitogram', 'fontSize', 16);

disp(size(single( effLBP(:) )))
%% Verify 'efficientLBP' and 'pixelwiseLBP' act alike, 
%% just with different run time and memory utilization
tic;
 % note this filter dimentions aren't legete...
effLBP= efficientLBP(imgRGB, 'filtR', filtR, 'isRotInv', true, 'isChanWiseRot', false);
effTime=toc;


% verify pixel wise implementation returns same results
tic;
% same parameters as before
pwLBP=pixelwiseLBP(imgRGB, 'filtR', filtR, 'isRotInv', true, 'isChanWiseRot', false);
inEffTime=toc;
fprintf('\nRun time ratio %.2f. Same result equality chesk: %o.\n', inEffTime/effTime,...
   isequal(effLBP, pwLBP));

figure;
subplot(1, 3, 1)
imshow(imgRGB);
title('Original image', 'fontSize', 18);

subplot(1, 3, 2)
imshow( effLBP );
title('Efficeint LBP image', 'fontSize', 18);

subplot(1, 3, 3)
imshow( pwLBP );
title('Pixel-wise LBP image', 'fontSize', 18);