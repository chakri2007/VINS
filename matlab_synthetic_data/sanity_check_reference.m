function sanity_check_reference(matFileReference)
%sanity_check_reference Re-runs estimateGravityRotationAndPoseScale on
%   the known-good reference dataset (vi_alignment_debug.mat, saved
%   directly by the original MATLAB .mlx run) using the SAME calling
%   convention as test_against_python.m, and checks whether it
%   reproduces the already-saved matlab_scale / matlab_gRot_R.
%
%   This validates the TEST HARNESS itself (argument construction,
%   se3/rigidtform3d usage, cell array handling) independently of any
%   question about your Python pipeline's data quality.
%
%   Usage:
%       sanity_check_reference('vi_alignment_debug.mat')

data = load(matFileReference);

N = size(data.campose_R, 3);

% ---- Camera poses ----
absPoses = repmat(rigidtform3d, N, 1);
for k = 1:N
    absPoses(k) = rigidtform3d(data.campose_R(:,:,k), data.campose_t(k,:));
end
camPoses = table(absPoses, 'VariableNames', {'AbsolutePose'});

% ---- Gyro/accel cells ----
gyro  = data.gyroData;
accel = data.accelData;

% ---- Sensor transform ----
T_BS = se3(double(data.T_BS_R), double(data.T_BS_t(:)'));

% ---- IMU parameters (note: noise fields here are 3x3 diagonal
%      matrices in this reference file, not scalars -- take the
%      diagonal value since factorIMUParameters expects a scalar or
%      3-vector, matching what the original .mlx recorded) ----
imuParams = factorIMUParameters( ...
    'SampleRate', double(data.imuSampleRate), ...
    'GyroscopeNoise', double(data.imuGyroNoise(1,1)), ...
    'GyroscopeBiasNoise', double(data.imuGyroBiasNoise(1,1)), ...
    'AccelerometerNoise', double(data.imuAccelNoise(1,1)), ...
    'AccelerometerBiasNoise', double(data.imuAccelBiasNoise(1,1)));

[gRot, scale, info] = estimateGravityRotationAndPoseScale( ...
    camPoses, gyro, accel, ...
    SensorTransform=T_BS, IMUParameters=imuParams);

fprintf('---- Freshly computed (this script) ----\n');
disp("scale: " + scale);
disp("gRot.A(1:3,1:3):"); disp(gRot.A(1:3,1:3));
disp("IsSolutionUsable: " + info.IsSolutionUsable);

fprintf('\n---- Saved reference (from original .mlx run) ----\n');
disp("matlab_scale: " + data.matlab_scale);
disp("matlab_gRot_R:"); disp(data.matlab_gRot_R);
disp("matlab_IsSolutionUsable: " + data.matlab_IsSolutionUsable);

fprintf('\n---- Diff ----\n');
scaleDiff = abs(scale - data.matlab_scale);
rotDiff = norm(gRot.A(1:3,1:3) - data.matlab_gRot_R, 'fro');
disp("Scale diff: " + scaleDiff);
disp("Rotation matrix Frobenius diff: " + rotDiff);

if scaleDiff < 1e-3 && rotDiff < 1e-3
    fprintf('\n[PASS] Test harness reproduces the saved reference answer closely.\n');
    fprintf('       -> Earlier mismatch on your pipeline data is a data/excitation issue, not a harness bug.\n');
else
    fprintf('\n[FAIL] Test harness does NOT reproduce the saved reference answer.\n');
    fprintf('       -> There is a real bug in how test_against_python.m calls the function.\n');
end

end