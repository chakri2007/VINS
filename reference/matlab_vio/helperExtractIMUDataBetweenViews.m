function [gyro,accel] = helperExtractIMUDataBetweenViews(gyroReadings,accelReadings,timeStamps,frameIds)
% helperExtractIMUDataBetweenViews extracts IMU Data (accelerometer and
% gyroscope readings) between specified consecutive frames.
%
%   This is an example helper function that is subject to change or removal 
%   in future releases.

%   Copyright 2023 The MathWorks, Inc.

len = length(frameIds);
gyro = cell(1,len-1);
accel = cell(1,len-1);
for k = 2:len
    % Assumes the IMU data is time-synchorized with the camera data. Compute
    % indices of accelerometer readings between consecutive view IDs.
    [~,ind1] = min(abs(timeStamps.imuTimeStamps - timeStamps.imageTimeStamps(frameIds(k-1))));
    [~,ind2] = min(abs(timeStamps.imuTimeStamps - timeStamps.imageTimeStamps(frameIds(k))));
    imuIndBetweenFrames = ind1:(ind2-1);
    % Extract the data at the computed indices and store in a cell.
    gyro{k-1} = gyroReadings(imuIndBetweenFrames,:);
    accel{k-1} = accelReadings(imuIndBetweenFrames,:);
end
end