function [currPoseRefineed,velRefined,biasRefined,valid] = helperBundleAdjustmentMotion(...
    xyzTrackedInCurrentView,currentViewCorrespondenses,intrinsics,imageSize, ...
    currentViewPoseGuess,currentViewVelocityGuess, previousViewPose, previousViewVelocity, previousViewBias, fIMU)
%helperBundleAdjustmentMotion computed refined pose, velocity and bias of
%   the current view with the help of factor graph optimization 
%
%   This is an example helper function that is subject to change or removal 
%   in future releases.

%   Copyright 2022 The MathWorks, Inc.

% add camera projection factor corrensponding to 3D-2D correspondenses
% found in current view
K = intrinsics.K;
f = factorGraph();
wPids = max(fIMU.NodeID) + ((1:size(currentViewCorrespondenses,1))');
fCam = factorCameraSE3AndPointXYZ([ones(size(xyzTrackedInCurrentView,1),1)*fIMU.NodeID(4),wPids], K, ...
    Information = ((K(1,1)/1.5)^2)*eye(2), Measurement=currentViewCorrespondenses);

f.addFactor(fCam);

% add IMU factor between previous and current views
f.addFactor(fIMU);

% fix XYZ points, previous view pose, velocity and bias during optimization
f.nodeState(fIMU.NodeID([1,4]),[previousViewPose;currentViewPoseGuess]);
f.nodeState(wPids,xyzTrackedInCurrentView);
f.nodeState(fIMU.NodeID([2,5]),[previousViewVelocity;currentViewVelocityGuess]);
f.nodeState(fIMU.NodeID([3,6]),[previousViewBias;previousViewBias]);
f.fixNode(wPids);
f.fixNode(fIMU.NodeID(1:3));

% add prior to current velocity and bias so that the optimization result is
% not too far away from the initial guess
velPriorFactorC = factorVelocity3Prior(fIMU.NodeID(5),Measurement=currentViewVelocityGuess);
addFactor(f,velPriorFactorC);

biasPriorFactorC = factorIMUBiasPrior(fIMU.NodeID(6),...
    Measurement=previousViewBias);
addFactor(f,biasPriorFactorC);

opts = factorGraphSolverOptions;
opts.MaxIterations = 10;
opts.VerbosityLevel = 0;
opts.FunctionTolerance  = 1e-5;
opts.GradientTolerance = 1e-5;
opts.StepTolerance = 1e-5;

% run factor graph optimization
optimize(f,opts);

% extract refined current view pose, velocity and bias
refinedVec = f.nodeState(fIMU.NodeID(4));
currPoseRefineed = rigidtform3d(quat2rotm(refinedVec(1,4:7)),refinedVec(1,1:3));

% check validity of matches w.r.t optimized current camera pose
ipts = [xyzTrackedInCurrentView,ones(size(xyzTrackedInCurrentView,1),1)]*(K*[currPoseRefineed.R',-(currPoseRefineed.R')*(currPoseRefineed.Translation')])';
imagePoints = ipts(:,1:2)./ipts(:,3);
valid = (ipts(:,3)>0)&(imagePoints(:, 1) >= 0)...
        & (imagePoints(:, 1) <= imageSize(2))...
        & (imagePoints(:, 2) >= 0)...
        & (imagePoints(:, 2) <= imageSize(1));

cc = (currentViewCorrespondenses - imagePoints);
valid = valid & (vecnorm(cc,2,2) < 5);

velRefined = f.nodeState(fIMU.NodeID(5));
biasRefined = f.nodeState(fIMU.NodeID(6));
end