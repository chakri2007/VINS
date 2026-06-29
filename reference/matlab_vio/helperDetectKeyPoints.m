function keyPoints = helperDetectKeyPoints(grayImage)
%helperDetectKeyPoints
%
%   This is an example helper function that is subject to change or removal 
%   in future releases.

%   Copyright 2023 The MathWorks, Inc.

% Detect multi-scale FAST corners.
keyPoints = detectORBFeatures(grayImage,ScaleFactor=1.2,NumLevels=4);
% Uncomment any of the following or try different detectors to tune
% keyPoints = detectFASTFeatures(grayImage,MinQuality=0.0786);
% keyPoints = detectMinEigenFeatures(grayImage,MinQuality=0.01,FilterSize=3);
end