function [posesUpdated,xyzUpdated] = helperTransformToNavigationFrame(poses,xyz,gRot,poseScale)
% helperTransformToNavigationFrame transforms and scales the input poses and XYZ points
% using specified gravity rotation and pose scale.
%
%   This is an example helper function that is subject to change or removal 
%   in future releases.

%   Copyright 2023 The MathWorks, Inc.

posesUpdated = poses;
% Input gravity rotation transforms the gravity vector from local 
% navigation reference frame to initial camera pose reference frame.
% The inverse of this transforms the poses from camera reference frame 
% to local navigation reference frame.
Ai = gRot.A';
for k = 1:length(poses.AbsolutePose)
    T = Ai*poses.AbsolutePose(k).A;
    T(1:3,4) = poseScale*T(1:3,4);
    posesUpdated.AbsolutePose(k) = rigidtform3d(T); 
end
% Transform points from initial camera pose reference frame to
% local navigation reference frame of IMU.
xyzUpdated = poseScale*gRot.transformPointsInverse(xyz);
end