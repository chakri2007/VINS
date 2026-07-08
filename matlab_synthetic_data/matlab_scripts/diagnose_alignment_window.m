function diagnose_alignment_window(matFilePython)
%diagnose_alignment_window Checks per-interval IMU sample counts and
%   inter-keyframe rotation/translation excitation, to spot degenerate
%   or under-excited intervals that can destabilize VI alignment.
%
%   Usage:
%       diagnose_alignment_window('vi_alignment_debug_python.mat')

data = load(matFilePython);

N = size(data.campose_R, 3);          % number of keyframes
numIntervals = N - 1;

fprintf('%-4s %-10s %-14s %-16s %-14s\n', ...
    'Idx','GyroN','AccelN','RotAngle(deg)','TransNorm(m)');
fprintf('%s\n', repmat('-', 1, 60));

rotAngles  = zeros(numIntervals,1);
transNorms = zeros(numIntervals,1);
gyroCounts  = zeros(numIntervals,1);
accelCounts = zeros(numIntervals,1);

for k = 1:numIntervals
    Ri = data.campose_R(:,:,k);
    Rj = data.campose_R(:,:,k+1);
    ti = data.campose_t(k,:);
    tj = data.campose_t(k+1,:);

    % relative rotation angle between consecutive keyframes
    Rrel = Ri' * Rj;
    cosAngle = (trace(Rrel) - 1) / 2;
    cosAngle = min(1, max(-1, cosAngle));  % clamp for numerical safety
    angleDeg = acosd(cosAngle);

    transNorm = norm(tj - ti);

    g = data.gyroData{1,k};
    a = data.accelData{1,k};

    rotAngles(k)   = angleDeg;
    transNorms(k)  = transNorm;
    gyroCounts(k)  = size(g,1);
    accelCounts(k) = size(a,1);

    fprintf('%-4d %-10d %-14d %-16.3f %-14.4f\n', ...
        k, gyroCounts(k), accelCounts(k), angleDeg, transNorm);
end

fprintf('\n---- Summary ----\n');
fprintf('Sample count  -> min: %d, max: %d, median: %.1f\n', ...
    min(accelCounts), max(accelCounts), median(accelCounts));
fprintf('Rotation(deg) -> min: %.3f, max: %.3f, median: %.3f\n', ...
    min(rotAngles), max(rotAngles), median(rotAngles));
fprintf('Translation   -> min: %.4f, max: %.4f, median: %.4f\n', ...
    min(transNorms), max(transNorms), median(transNorms));

% Flag suspicious intervals
lowSampleThresh = 10;   % tune as needed
lowRotThresh    = 0.5;  % degrees, tune as needed

flaggedSamples = find(accelCounts < lowSampleThresh);
flaggedRot     = find(rotAngles < lowRotThresh);

if ~isempty(flaggedSamples)
    fprintf('\n[WARNING] Intervals with < %d IMU samples: %s\n', ...
        lowSampleThresh, mat2str(flaggedSamples'));
end
if ~isempty(flaggedRot)
    fprintf('[WARNING] Intervals with < %.2f deg rotation (low excitation): %s\n', ...
        lowRotThresh, mat2str(flaggedRot'));
end
if isempty(flaggedSamples) && isempty(flaggedRot)
    fprintf('\nNo obviously degenerate intervals found by these thresholds.\n');
end

end