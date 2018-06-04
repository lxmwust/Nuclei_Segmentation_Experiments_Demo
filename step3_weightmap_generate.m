%==========================================================================
% This code is used to generate interval, masker and  weight map for
% training network
%-------------------------------------------------------------------------
% Author:Lipeng Xie
% Date:2017-07-28
%==========================================================================
clc;
clear;
close all;
addpath ./tools


%% ===============setp1: set parameters==============
DatasetName ='BNS'; %BNS or MICCAI2017
datapath = ['./data/' DatasetName '/'];
num_DB = 1; %number of databases
patchfolder = 'patches'; %fold name of samples 
for k=1:num_DB
    save_to_dir = [datapath patchfolder '_' int2str(k) '/'];
    files = dir([save_to_dir  '*_l.png']);%ÐÞ¸Äºó×º % we only use images for which we have a mask available, so we use their filenames to limit the OriginImgNames
    for i=1:length(files)
        filename = regexp(files(i).name,'_l', 'split');
        filename = filename{1};
        mask = imread([save_to_dir files(i).name]);
        
        %================foreground_weight=============================
        Foreground_weight = WeightDataGenerator_classbalance(mask);
        save([save_to_dir filename '_fw.mat'],'Foreground_weight');
        
        %================Interval_weight=============================
        [Interval_mask, Interval_weight]  = IntervalGenerator(mask,0.4);
        imwrite(Interval_mask,[save_to_dir filename '_im.png']); 
        save([save_to_dir filename '_iw.mat'],'Interval_weight');

        %================Masker_weight=============================
        [Masker_mask, Masker_weight]  = MaskerGenerator(mask,2);
        imwrite(Masker_mask,[save_to_dir filename '_mm.png']);     
        save([save_to_dir filename '_mw.mat'],'Masker_weight');

    end

end