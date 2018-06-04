function [score,precision,recall]= F1score(S,G)
% F1score calculates f1 score for detection
%
% Inputs:
%   S: a label image contains segmented objects
%   G: a label image contains ground truth annotation
%
% Outputs:
%   score: f1 score
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

listS = unique(S);            % list of labels of segmented objects
listS(listS == 0) = [];       % remove the label of the background
numS = length(listS);         % the total number of segmented objects in S

listG = unique(G);            % list of labels of ground truth objects
listG(listG == 0) = [];       % remove the label of the background
numG = length(listG);         % the total number of ground truth objects in G

if numS == 0 && numG == 0    % no segmented object & no ground truth objects
    score = 1;
    return 
elseif numS == 0 || numG == 0
    score = 0;
    return
else
    % do nothing
end

% Identify a corresponding ground truth object in G for each segmented
% object in S

tempMat = zeros(numS,3);     % a temporary matrix
tempMat(:,1) = listS;        % the 1st col contains labels of segmented objects
                            % the 2nd col contains labels of the
                            % corresponding ground truth objects
                            % the 3rd col contains true positive flags

for iSegmentedObj = 1:numS
    intersectGTObjs = G(S == tempMat(iSegmentedObj,1));
    intersectGTObjs(intersectGTObjs == 0) = [];
    if ~isempty(intersectGTObjs)
        listOfIntersectGTObjs = unique(intersectGTObjs);
        N = histc(intersectGTObjs,listOfIntersectGTObjs);
        [~,maxId] = max(N);
        tempMat(iSegmentedObj,2) = listOfIntersectGTObjs(maxId);
    end
end

% Identify true positives, false positives, false negatives
for iSegmentedObj = 1:numS
    if tempMat(iSegmentedObj,2) ~= 0            % avoid false positive objects that only intersect with background
        SegObj = S == tempMat(iSegmentedObj,1);
        GTObj = G == tempMat(iSegmentedObj,2);
        overlap = SegObj & GTObj;
        areaOverlap = sum(overlap(:));
        areaGTObj = sum(GTObj(:));
        if areaOverlap/areaGTObj > 0.5
            tempMat(iSegmentedObj,3) = 1;       % flag for true positive
        end
    end
end

TP = sum(tempMat(:,3) == 1);
FP = sum(tempMat(:,3) == 0);

listOfGTObjs = unique(G);
listOfGTObjs(listOfGTObjs == 0) = [];
listOfGTObjsWithCorrespondingSegObj = tempMat(:,2);
listOfGTObjsWithCorrespondingSegObj(listOfGTObjsWithCorrespondingSegObj == 0) = [];
listOfGTObjsWithOutCorrespondingSegObj = setdiff(listOfGTObjs,listOfGTObjsWithCorrespondingSegObj);

FN = length(listOfGTObjsWithOutCorrespondingSegObj);

precision = TP/(TP + FP);
recall = TP/(TP + FN);
score = (2*precision*recall)/(precision+recall);
end
