function [] = extractLBP()
imgRGB = imread('F:\cse802_data\lfw_mtcnn_cropped\LFW_Aaron-Eckhart\LFW_Aaron_Eckhart_0001.jpg');
img = rgb2gray(imgRGB);
lbpfeature = extractLBPFeatures(img,'Upright',false);
disp(size(lbpfeature))

end