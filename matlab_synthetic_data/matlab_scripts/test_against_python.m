function test_against_python(matFilePython, matFileResult)
%test_against_python Loads Python's pre-VI-alignment debug dump and runs
%   MATLAB's estimateGravityRotationAndPoseScale on the exact same
%   inputs, then compares against Python's own recovered scale/gravity.
%
%   Usage:
%       test_against_python('vi_alignment_debug_python.mat', ...
%                            'vi_alignment_debug_python_result.mat')

if nargin < 2
    matFileResult = strrep(matFilePython, '.mat', '_result.mat');
end

data = load(matFilePython);

N = size(data.campose_R, 3);

% Reconstruct camera poses as rigidtform3d array
absPoses = repmat(rigidtform3d, N, 1);
for k = 1:N
    absPoses(k) = rigidtform3d(data.campose_R(:,:,k), data.campose_t(k,:));
end
camPoses = table(absPoses, 'VariableNames', {'AbsolutePose'});

% Reconstruct gyro/accel cell arrays (already 1x(N-1) cells from scipy)
gyro = data.gyroData;
accel = data.accelData;

% Camera->IMU extrinsic
T_BS = se3(data.T_BS_R, data.T_BS_t(:)');

% IMU parameters
imuParams = factorIMUParameters( ...
    'SampleRate', double(data.imuSampleRate), ...
    'GyroscopeNoise', double(data.imuGyroNoise), ...
    'GyroscopeBiasNoise', double(data.imuGyroBiasNoise), ...
    'AccelerometerNoise', double(data.imuAccelNoise), ...
    'AccelerometerBiasNoise', double(data.imuAccelBiasNoise));

[gRot, scale, info] = estimateGravityRotationAndPoseScale( ...
    camPoses, gyro, accel, ...
    SensorTransform=T_BS, IMUParameters=imuParams);

disp("MATLAB estimated scale: " + scale);
disp("MATLAB gravity rotation A:"); disp(gRot.A);
disp("MATLAB IsSolutionUsable: " + info.IsSolutionUsable);

if isfile(matFileResult)
    py = load(matFileResult);
    disp("---- Python result ----");
    disp("Python success: " + py.python_success);
    disp("Python scale:   " + py.python_scale);
    disp("Python gravity: "); disp(py.python_gravity);
    disp("Python accel bias: "); disp(py.python_accel_bias);

    disp("---- Diff ----");
    disp("Scale diff: " + abs(scale - py.python_scale));
else
    warning("Result file %s not found -- skipping Python comparison.", matFileResult);
end
end