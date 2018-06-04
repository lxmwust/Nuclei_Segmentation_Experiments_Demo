%==========================================================================
% This functiom is used to generate weight map for image
%-------------------------------------------------------------------------
% Author:Lipeng Xie
% Date:2017-08-08
%==========================================================================
function Weight_final = WeightDataGenerator_classbalance(mask)
%% step1: Get edge
[H,W] = size(mask);
%% step42: compute the class weight
Num1 = sum(mask(:));
TotalNum = H*W;
Num2 = TotalNum - Num1;
Weight_class = mask.*(Num2/TotalNum) + (~mask).*(Num1/TotalNum);
%% step5: combine weight
Weight_final = Weight_class;
