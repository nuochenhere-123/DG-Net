clc;clear all;
%% para_set
%% for ten enhanced images of one raw image:groups = 1; numbers_in_each_group = 10;
%% for ten enhanced images of ten raw image:groups = 10; numbers_in_each_group = 1;
groups = 45;
numbers_in_each_group = 1;
%% feature extraction
folder = 'D:\Desktop\test\WaterNet-ReTrain-U45-256';
files = dir(fullfile(folder,'*.png'));
%% files = dir(fullfile(folder,'*.jpg'));
numFiles = length(files);

file_names = cell(numFiles, 1); % 用于保存文件名

for i=1:numFiles
    % 打印当前迭代的索引值
    disp(['Processing file ' num2str(i) ' of ' num2str(numFiles)]);
    
    filename = files(i).name;
    fullname = fullfile(folder,filename);
    img = imread(fullname);
    final_feature(i,:) = NUIQ_feature_extraction(img);
    % 保存文件名
    file_names{i} = filename;
    
end
final_feature = final_feature';
final_feature = mapminmax(final_feature,0,1);
final_feature = final_feature';
%% score prediction
predictions = test_model(final_feature,groups,numbers_in_each_group,file_names);

% 显示或保存带有文件名的预测结果
% results_table = table(file_names, predictions);
% disp(results_table);

% 可选：保存结果为 CSV 文件
% output_filename = fullfile(folder, 'prediction_results.csv');
% writetable(results_table, output_filename);
