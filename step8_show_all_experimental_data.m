close all;
clear all;

%% ======================Set parameters==================== 
DatasetName ='BNS'; %BNS or MICCAI2017
NetworkList = {'FCN8s','HED','Unet','SharpMask','DIMAN'};% 'FCN8s','HED','Unet','SharpMask','DIMAN'
%===================================================
switch DatasetName
        case 'MICCAI2017'    
            oritImgpath = '../Nuclei_segmentation_testing/';
        case 'BNS'
            oritImgpath = '../BNS_Nuclei_Data/';
        otherwise
            error('Unknown Dataset.')
end
%'fg','fg_watershed','fg_dynamicwatershed','fg_erosionwatershed','refine','refine_watershed','refine_dynamicwatershed','refine_erosionwatershed','fg_interval_marker_watershed'

Alldata_mean =[];
Alldata_std = []; 
for j=1:length(NetworkList)
    NetworkName = NetworkList{j};
    if strcmp(NetworkName,'DIMAN')
        Evaluation_items = {'fg','refine','refine_watershed','refine_erosionwatershed','refine_dynamicwatershed','fg_interval_marker_watershed'};
    else
        Evaluation_items = {'fg','fg_watershed','fg_erosionwatershed','fg_dynamicwatershed'};
        
    end
    ResultPath = ['./outputs/' DatasetName '/' NetworkName '/'];
 
    %% ===================evaluation=================================
   
    for ii=1:length(Evaluation_items)
        item = Evaluation_items{ii};
        load([ResultPath 'EvaluScore_' item '.mat']); 
        Alldata_mean = [Alldata_mean;mean(EvaluScore)];
        Alldata_std = [Alldata_std;std(EvaluScore)];
    end


end
    