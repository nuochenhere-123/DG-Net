clc;
clear all;

% Fu X, Zhuang P, Huang Y, et al. 
% A retinex-based enhancing approach for single underwater image
% [C]//2014 IEEE International Conference on Image Processing (ICIP). IEEE, 2014: 4572-4576.

sc = 1;
folder2='.\ImageForTest\Test-110';
filepaths = dir(fullfile(folder2,'*.png'));
% output_folder = '.\RetinexBased256\RetinexBased-110\';
output_folder = '.\RetinexBased\RetinexBased-110\';
if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

for ii= 1 : length(filepaths)
    tic
   
input = im2double(imread(fullfile(folder2,filepaths(ii).name)));
% input = imresize(input, [256,256]);

output = underwater(input);
imwrite(output,fullfile(output_folder,filepaths(ii).name));
end