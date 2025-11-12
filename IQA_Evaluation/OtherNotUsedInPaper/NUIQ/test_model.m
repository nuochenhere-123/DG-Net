function [predictions_res] = test_model(feature,groups,numbers_in_each_group,file_names)
%% test model
test_data = feature;
% test_label = d(801:1000,:);
test_label = zeros(groups * numbers_in_each_group,1);

fid2=fopen('E:\Codefield\Matlab\NUIQ\test.dat','w');
dat_creat(test_label,test_data,fid2,59,groups,numbers_in_each_group);
fclose(fid2);

system('svm_rank_classify.exe test.dat model predictions');
load predictions

% 将文件名和预测结果组合成一个表格
predictions_res = table(file_names, predictions);
end

