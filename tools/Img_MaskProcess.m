%==========================================================================
% There are some errors existing in the original Mask data. 
% For example, two different objects have the same label.
% This code is used to process the original Mask.
%-------------------------------------------------------------------------
% Author:Lipeng Xie
% Date:2017-08-7
%==========================================================================
function Mask = Img_MaskProcess(Mask_ori)
Maxlabel = max(Mask_ori(:));
Mask = zeros(size(Mask_ori));
Num = 0;
for i=1:Maxlabel
    M = Mask_ori==i;
    M_sup = M*Num;
    M = bwlabel(M>0);
    Mask = Mask + M + M_sup;
    Num = Num + max(M(:));
end
