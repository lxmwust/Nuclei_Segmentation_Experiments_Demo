%==========================================================================
% This code is used to process the mask by spliting the overlap cells.
%-------------------------------------------------------------------------
% Author:Lipeng Xie
% Date:2017-08-28
%==========================================================================
function mask_final = Mask_split_overlap(mask_ori)
mask_ori = mask_ori(:,:,1);
[H,W] = size(mask_ori);

mask_adja = zeros(H,W,'uint8');
parfor i=2:H-1
    for j=2:W-1
        if mask_ori(i,j)==0
            continue
        end
        temp = mask_ori (i-1:i+1,j-1:j+1);
        temp = temp(:);
        index= find(temp>0);
        temp = temp(index);
        if length(unique(temp))>=2
            mask_adja(i,j) =1;
        end
    end
end

mask_final = uint8(mask_ori>0) - mask_adja;
