function test_against_python_trimmed(matFilePython, matFileResult, dropFirstN)
%test_against_python_trimmed Re-runs MATLAB's
%   estimateGravityRotationAndPoseScale after dropping the first
%   dropFirstN keyframes (default 1) to remove degenerate/near-zero-
%   motion intervals, then compares against the full-window result and
%   Python's saved result.
%
%   Usage:
%       test_against_python_trimmed('vi_alignment_debug_python.mat')
%       test_against_python_trimmed('vi_alignment_debug_python.mat', ...
%           'vi_alignment_debug_python_result.mat', 1)

if nargin < 3
    dropFirstN = 1;
end
if nargin < 2 || isempty(matFileResult)
    matFileResult = strrep(matFilePython, '.mat', '_result.mat');
end

data = load(matFilePython);
N = size(data.campose_R, 3);

if dropFirstN >= N - 2
    error('dropFirstN too large -- not enough keyframes left to align.');
end

keepIdx = (dropFirstN + 1):N;

% ---- Trim camera poses ----
absPoses = repmat(rigidtform3d, numel(keepIdx), 1);
for kk = 1:numel(keepIdx)
    k = keepIdx(kk);
    absPoses(kk) = rigidtform3d(data.campose_R(:,:,k), data.campose_t(k,:));
end
camPoses = table(absPoses, 'VariableNames', {'AbsolutePose'});

% ---- Trim gyro/accel cells (interval k is between keyframe k and k+1,
%      so dropping the first dropFirstN keyframes also drops the first
%      dropFirstN intervals) ----
gyro  = data.gyroData(1, (dropFirstN + 1):end);
accel = data.accelData(1, (dropFirstN + 1):end);

% ---- Sensor transform / IMU params (unchanged) ----
T_BS = se3(data.T_BS_R, data.T_BS_t(:)');

imuParams = factorIMUParameters( ...
    'SampleRate', double(data.imuSampleRate), ...
    'GyroscopeNoise', double(data.imuGyroNoise), ...
    'GyroscopeBiasNoise', double(data.imuGyroBiasNoise), ...
    'AccelerometerNoise', double(data.imuAccelNoise), ...
    'AccelerometerBiasNoise', double(data.imuAccelBiasNoise));

[gRot, scale, info] = estimateGravityRotationAndPoseScale( ...
    camPoses, gyro, accel, ...
    SensorTransform=T_BS, IMUParameters=imuParams);

fprintf('---- Trimmed window (dropped first %d keyframe(s)) ----\n', dropFirstN);
disp("MATLAB (trimmed) scale: " + scale);
disp("MATLAB (trimmed) gravity rotation A:"); disp(gRot.A);
disp("MATLAB (trimmed) IsSolutionUsable: " + info.IsSolutionUsable);

% ---- Also re-run full window for a side-by-side reference ----
absPosesFull = repmat(rigidtform3d, N, 1);
for k = 1:N
    absPosesFull(k) = rigidtform3d(data.campose_R(:,:,k), data.campose_t(k,:));
end
camPosesFull = table(absPosesFull, 'VariableNames', {'AbsolutePose'});

[gRotFull, scaleFull, infoFull] = estimateGravityRotationAndPoseScale( ...
    camPosesFull, data.gyroData, data.accelData, ...
    SensorTransform=T_BS, IMUParameters=imuParams);

fprintf('\n---- Full window (for reference) ----\n');
disp("MATLAB (full) scale: " + scaleFull);
disp("MATLAB (full) IsSolutionUsable: " + infoFull.IsSolutionUsable);

if isfile(matFileResult)
    py = load(matFileResult);
    fprintf('\n---- Python result (full window) ----\n');
    disp("Python scale: " + py.python_scale);
    disp("Python gravity:"); disp(py.python_gravity);

    fprintf('\n---- Diffs ----\n');
    disp("Trimmed vs Python scale diff: " + abs(scale - py.python_scale));
    disp("Full    vs Python scale diff: " + abs(scaleFull - py.python_scale));
else
    warning("Result file %s not found -- skipping Python comparison.", matFileResult);
end

end