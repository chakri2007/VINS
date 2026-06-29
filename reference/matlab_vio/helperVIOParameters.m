function [params] = helperVIOParameters()
%helperVIOParameters helper to initialize Visual-Inertial odometry
%   algorithm parameters
%
%   This is an example helper function that is subject to change or removal 
%   in future releases.

%   Copyright 2022-2023 The MathWorks, Inc.

% factor graph solver options
opts = factorGraphSolverOptions;
opts.MaxIterations = 5;
opts.VerbosityLevel = 0;
opts.FunctionTolerance  = 1e-5;
opts.GradientTolerance = 1e-5;
opts.StepTolerance = 1e-5;
opts.TrustRegionStrategyType = 1;
params.SolverOpts =opts;

% sliding window size - maximum number of frames in sliding window.
params.windowSize = 21;
% minimum number of key point tracks in a frame
params.numTrackedThresh = 100;
% maxmium number of key points in a frame
params.maxPointsToTrack = 150;

% set to true if the images contain perspective distortion
params.Undistort = false;

% set to true to enhance the contrast of images captured in dark
% environments
params.Equalize = false;
% important equalization parameters
params.ClipLimit = 3/256;
params.NumTiles = [8,8];

% KLT tracker parameters
params.KLT_BiErr = 1; % bidirectional error
params.KLT_Levels = 4;
params.KLT_Block = [21,21]; % block size

% RANSAC parameters
params.F_Threshold = 4; % random sample consensus threshold
params.F_Confidence = 99;
params.F_Iterations = 2000;
params.F_loop = 5;

% parallax in number of pixels 
params.keyFrameParallax = 50; % minimum parallax for key frame selection 
params.triangulateParallax = 30; % minimum parallax for triangulating new 3-D points

% run sliding window optimization after every few frames to reduce
% computational time
params.optimizationFrequency = 3;
end