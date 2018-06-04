%==========================================================================
% This code is used to cut the images into many pieces for training and
% testing.
%-------------------------------------------------------------------------
% Author:Lipeng Xie
% Date:2017-03-20
%==========================================================================
clear;
clc;

addpath ./tools
%% ===============setp1: set parameters==============
DatasetName ='BNS'; %BNS or MICCAI2017
switch DatasetName
        case 'MICCAI2017'    
            datapath_images = './dataset/MICCAI2017/Nuclei_segmentation_training/';
            subfoldname_images = dir(datapath_images);
        case 'BNS'
            datapath_images = './dataset/BNS/BNS_Nuclei_Data/';
            subfoldname_images  = dir([datapath_images,'Slide_*']);
        otherwise
            error('Unknown Dataset.')
end

datapath = ['./data/' DatasetName '/'];
if ~exist(datapath,'dir')
    mkdir(datapath);
end
num_DB = 1; %number of databases
patchfolder = 'patches'; %fold name of samples 

%----------------images augmentation parameters------------------
para_imgaug.maxnum = 1;
para_imgaug.cropsize = 224; %size of the pathces
para_imgaug.random_fliplr = true;
para_imgaug.random_flipup = true;
para_imgaug.random_dropout = false; % drop rate 0~1
para_imgaug.save_format = '.png';
%% ===============setp2: get the list of images==============
for k=1:num_DB 
    save_to_dir = [datapath patchfolder '_' int2str(k) '/'];
    if ~exist(save_to_dir,'dir')
        mkdir(save_to_dir);
    end
    parfor i=1:length(subfoldname_images)
        if strcmp(subfoldname_images(i).name,'.') || strcmp(subfoldname_images(i).name,'..')
            continue;
        end
        files = dir([datapath_images subfoldname_images(i).name  '/training-set/' '*_mask.png']);% we only use images for which we have a mask available, so we use their filenames to limit the OriginImgNames
        OriginImgNames = unique(arrayfun(@(x) x{1}{1},arrayfun(@(x) regexp(x.name,'_mask', 'split'),files,'UniformOutput',0),'UniformOutput',0)); %this creates a list of patient id numbers
        for j=1:length(OriginImgNames)
            save_prefix = [subfoldname_images(i).name '_' OriginImgNames{j}];
            Img = imread([datapath_images subfoldname_images(i).name  '/training-set/' OriginImgNames{j} '.png']);
            mask = imread([datapath_images subfoldname_images(i).name  '/training-set/' OriginImgNames{j} '_mask.png']);
            mask = Mask_split_overlap(mask);
            ImageDataGenerator_withmask(Img,mask,save_prefix,save_to_dir,para_imgaug);
        end       
   end    

end
   


