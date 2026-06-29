function I = helperProcessImage(I,params,intrinsics)
%helperProcessImage Equalize and undistort images if needed
%
%   This is an example helper function that is subject to change or removal 
%   in future releases.

%   Copyright 2023 The MathWorks, Inc.

if params.Equalize
    % Enhance contrast if images are dark.
    I = adapthisteq(I,NumTiles=params.NumTiles,ClipLimit=params.ClipLimit);
end
if params.Undistort
    % Undistort if images contain perspective distortion.
    I = undistortImage(I,intrinsics);
end
end