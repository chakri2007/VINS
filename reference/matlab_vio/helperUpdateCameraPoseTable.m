function cameraPoseTableUpdated = helperUpdateCameraPoseTable(cameraPoseTable,cameraPoses)
% helperUpdateCameraPoseTable updates camera pose table with specified
% N-by-7 SE(3) camera poses.
%
%   This is an example helper function that is subject to change or removal 
%   in future releases.

%   Copyright 2023 The MathWorks, Inc.

cameraPoseTableUpdated = cameraPoseTable;
R = quat2rotm(cameraPoses(:,4:7));
for k = 1:size(cameraPoses,1)
    cameraPoseTableUpdated.AbsolutePose(k).Translation = cameraPoses(k,1:3);
    cameraPoseTableUpdated.AbsolutePose(k).R = R(:,:,k);
end
end