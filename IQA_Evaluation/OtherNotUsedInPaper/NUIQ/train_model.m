clc;clear all;
%% load features and labels for training
load sub_votes.mat
load train_feats.mat
%% preprocessing
train_feats = train_feats';
train_feats = mapminmax(train_feats,0,1);
train_feats = train_feats';

train_data = train_feats;
train_label = sub_votes;
%% begin training
fid1=fopen('D:\Work_one_method\NUIQ_Metric\train.dat','w');
dat_creat(train_label,train_data,fid1,59,100,10);
fclose(fid1);
system('svm_rank_learn.exe -c 20 train.dat model');
answer = 'finish training';
disp(answer)