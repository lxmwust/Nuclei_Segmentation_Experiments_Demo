function objDice = ObjectDice(S,G)
% ObjectDice calculates an object-level Dice index 
% 
% Inputs:
%   S: a label image contains segmented objects, meaning that each object
%      is label with different unique integer number, and the background is
%      label by 0.
%   G: a label image contains ground truth objects, meaning that each object
%      is label with different unique integer number, and the background is
%      label by 0.
%   
% Outpus:
%   objDice: an object-level Dice index
%
%
% Korsuk Sirinukunwattana 
% BIAlab, Department of Computer Science, University of Warwick
% 2015
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% convert S and G to the same format
% BW=im2bw(S);
% S = bwlabel(S,8);
% BW=im2bw(G);
% G = bwlabel(BW,8);

S = single(S);
G = single(G);

% check if S or G is non-empty

listLabelS = unique(S);             % a list of labels of objects in S
listLabelS(listLabelS == 0) = [];
numS = length(listLabelS);

listLabelG = unique(G);             % a list of labels of objects in G
listLabelG(listLabelG == 0) = [];
numG = length(listLabelG);


if numS == 0 && numG == 0    % no segmented object & no ground truth objects
    objDice = 1;
    return 
elseif numS == 0 || numG == 0
    objDice = 0;
    return
else
    % do nothing
end

% calculate object-level dice
temp1 = 0;                          % omega_i*Dice(G_i,S_i)
totalAreaS = sum(S(:)>0);
for iLabelS = 1:length(listLabelS)
    Si = S == listLabelS(iLabelS);
    intersectlist = G(Si);
    intersectlist(intersectlist == 0) = [];
    
    if ~isempty(intersectlist)
        indexGi = mode(intersectlist);
        Gi = G == indexGi;
    else
        Gi = false(size(G));
    end
    
    omegai = sum(Si(:))/totalAreaS;
    temp1 = temp1 + omegai*Dice(Gi,Si);
end

temp2 = 0;                          % tilde_omega_i*Dice(tilde_G_i,tilde_S_i)
totalAreaG = sum(G(:)>0);
for iLabelG = 1:length(listLabelG)
    tildeGi = G == listLabelG(iLabelG);
    intersectlist = S(tildeGi);
    intersectlist(intersectlist == 0) = [];
    
    if ~isempty(intersectlist)
        indextildeSi = mode(intersectlist);
        tildeSi = S == indextildeSi;
    else
        tildeSi = false(size(S));
    end
    
    tildeOmegai = sum(tildeGi(:))/totalAreaG;
    temp2 = temp2 + tildeOmegai*Dice(tildeGi,tildeSi);
end

objDice = (temp1 + temp2)/2;

    function dice = Dice(A,B)
        temp = A&B;
        dice = 2*sum(temp(:))/(sum(A(:))+sum(B(:)));
    end
end


