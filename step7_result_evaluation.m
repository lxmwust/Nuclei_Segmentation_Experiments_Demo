close all;
clear all;

addpath './Evaluation Metrics'
addpath './tools'
%% ======================Set parameters==================== 
DatasetName ='BNS'; %BNS or MICCAI2017
NetworkList = {'FCN8s','HED','Unet','SharpMask','DIMAN'};% 'FCN8s','HED','Unet','SharpMask','DIMAN'
%===================================================
switch DatasetName
        case 'MICCAI2017'    
            oriImgpath = './dataset/MICCAI2017/Nuclei_segmentation_testing/';
        case 'BNS'
            oriImgpath = './dataset/BNS/BNS_Nuclei_Data/';
        otherwise
            error('Unknown Dataset.')
end
%'fg','fg_watershed','fg_dynamicwatershed','fg_erosionwatershed','refine','refine_watershed','refine_dynamicwatershed','refine_erosionwatershed','fg_interval_marker_watershed'
for j=1:length(NetworkList)
    NetworkName = NetworkList{j};
    if strcmp(NetworkName,'DIMAN')
        Evaluation_items = {'fg','refine','refine_watershed','refine_dynamicwatershed','refine_erosionwatershed','fg_interval_marker_watershed'};
    else
        Evaluation_items = {'fg','fg_watershed','fg_dynamicwatershed','fg_erosionwatershed'};
        
    end
    ResultPath = ['./outputs/' DatasetName '/' NetworkName '/'];
 
    %% ===================evaluation=================================
    for ii=1:length(Evaluation_items)
        item = Evaluation_items{ii};
        Imgnamelist = dir([ResultPath '*_' item '_result.mat']); 
        NumImg = length(Imgnamelist);
        EvaluScore = zeros(NumImg,5);
        for i=1:NumImg
            if mod(i,5)==0
               fprintf '.'
            end
            load([ResultPath Imgnamelist(i).name]);
            eval(['Img_result =' item '_result;']);
            imagenametemp = regexp(Imgnamelist(i).name,'_','split');
            if strcmp(DatasetName,'MICCAI2017')
                imagepath = imagenametemp{1};
                imagename = imagenametemp{2};
            else
                imagepath = [imagenametemp{1} '_' imagenametemp{2}];
                imagename = [imagenametemp{2} '_' imagenametemp{3}];
            end

            Img_mask = imread([oriImgpath imagepath '/testing-set/' imagename '_mask.png']);
            Img_mask = Img_mask(:,:,1);
            if strcmp(DatasetName,'BNS')
                Img_mask = Img_MaskProcess(Img_mask);
            end
            [score,precision,recall]= F1score(Img_result,Img_mask);
            EvaluScore(i,1:3)=[score,precision,recall];

            objDice = ObjectDice(Img_result,Img_mask);
            EvaluScore(i,4) = objDice;
            
            objectHausdorff = ObjectHausdorff(Img_result,Img_mask);
            EvaluScore(i,5) = objectHausdorff;
        end
        mean_F1 = mean(EvaluScore(:,1));
        mean_Dice = mean(EvaluScore(:,4));
        eval(['EvaluScore_Mwan_' item '= mean(EvaluScore);']);
        eval(['EvaluScore_Standard_' item '= std(EvaluScore);']);
        disp(sprintf('NetworkName: %s, Item: %s',NetworkName, item))
        disp(sprintf('mean_F1:%f,mean_Dice:%f ',mean_F1,mean_Dice))
        save([ResultPath 'EvaluScore_' item '.mat'],'EvaluScore');

    end


end
    