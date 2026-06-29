function [currPose,ii] = helperEstimateCameraPose(params,c2D,x3D,intrinsics,currPose,ii)
% helperEstimateCameraPose computes the camera pose
%
%   This is an example helper function that is subject to change or removal 
%   in future releases.

%   Copyright 2023 The MathWorks, Inc.

for k = 1:params.F_loop
    [currPosel,iil] = estworldpose( ...
        c2D,x3D, ...
        intrinsics,MaxReprojectionError=params.F_Threshold,Confidence=params.F_Confidence, ...
        MaxNumTrials=params.F_Iterations);
    if length(find(ii)) < length(find(iil))
        ii = iil;
        currPose = currPosel;
    end
end

end