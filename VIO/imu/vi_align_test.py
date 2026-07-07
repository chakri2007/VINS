"""
test_against_matlab.py

Loads vi_alignment_debug.mat (saved from the MATLAB reference example
right before its estimateGravityRotationAndPoseScale call) and feeds
the *exact same* camera poses + gyro/accel windows into the Python
vi_alignment module, in isolation from the rest of the VIO pipeline.

Usage:
    python imu/test_against_matlab.py /path/to/vi_alignment_debug.mat
"""

import sys
import numpy as np
from scipy.io import loadmat

from memory_management.view_set import ViewSet
from memory_management.sliding_window import SlidingWindowState
from imu.imu_measurement import IMUMeasurement
from imu.preintegration import IMUPreintegrator
from imu.vi_alignment import initialize_visual_inertial_state


def build_preintegration(gyro_block, accel_block, dt, bias_g, bias_a,
                          gyro_noise, accel_noise,
                          gyro_rw=1.0e-5, accel_rw=1.0e-4):
    """
    MATLAB's helperExtractIMUDataBetweenViews returns raw Mx3 gyro/accel
    blocks with no explicit per-sample timestamps -- estimateGravity...
    assumes a fixed IMU SampleRate. We reconstruct synthetic timestamps
    at that rate so IMUPreintegrator.integrate_measurements sees the
    same dt spacing MATLAB's factorIMU used internally.
    """
    n = gyro_block.shape[0]
    t0 = 0.0
    samples = [
        IMUMeasurement(
            timestamp=t0 + i * dt,
            accel=np.asarray(accel_block[i], dtype=np.float64),
            gyro=np.asarray(gyro_block[i], dtype=np.float64),
        )
        for i in range(n)
    ]
    preintegrator = IMUPreintegrator(
        gyro_noise=gyro_noise,
        accel_noise=accel_noise,
        gyro_random_walk=gyro_rw,
        accel_random_walk=accel_rw,
        bias_g=bias_g,
        bias_a=bias_a,
    )
    return preintegrator.integrate_measurements(samples)


def main(mat_path):
    data = loadmat(mat_path, squeeze_me=True, struct_as_record=False)

    sw_ids       = np.atleast_1d(data['swIDs']).astype(int).tolist()
    campose_R    = data['campose_R']     # (3,3,N)
    campose_t    = data['campose_t']     # (N,3)
    gyro_cells   = np.atleast_1d(data['gyroData'])
    accel_cells  = np.atleast_1d(data['accelData'])
    T_BS_R       = data['T_BS_R']
    T_BS_t       = data['T_BS_t']
    print("DEBUG - imuGyroNoise raw type/shape:", type(data['imuGyroNoise']), np.shape(data['imuGyroNoise']))
    print("DEBUG - imuGyroNoise values:", data['imuGyroNoise'])

    # Extract the top-left element from the covariance matrix diagonal
    imu_sample_rate     = float(np.ravel(data['imuSampleRate'])[0])
    imu_gyro_noise      = float(data['imuGyroNoise'][0, 0])
    imu_gyro_bias_noise = float(data['imuGyroBiasNoise'][0, 0])
    imu_accel_noise     = float(data['imuAccelNoise'][0, 0])
    imu_accel_bias_noise = float(data['imuAccelBiasNoise'][0, 0])

    matlab_scale = float(np.atleast_1d(data['matlab_scale'])[0])
    matlab_is_usable = bool(np.atleast_1d(data['matlab_IsSolutionUsable'])[0])

    dt = 1.0 / imu_sample_rate

    print(f"Loaded {len(sw_ids)} views, MATLAB scale={matlab_scale:.6g}, "
          f"IsSolutionUsable={matlab_is_usable}")

    # --- Build ViewSet with the exact MATLAB poses ---
    view_set = ViewSet()
    N = len(sw_ids)
    for k in range(N):
        R = campose_R[:, :, k]
        t = campose_t[k, :]
        view_set.add_view(sw_ids[k], R, t, timestamp=float(k) * dt)

    # --- Build a PreintegratedIMU per consecutive pair ---
    imu_preintegrations = {}
    for k in range(N - 1):
        gyro_block = np.atleast_2d(gyro_cells[k])
        accel_block = np.atleast_2d(accel_cells[k])
        preint = build_preintegration(
            gyro_block, accel_block, dt,
            bias_g=np.zeros(3), bias_a=np.zeros(3),
            gyro_noise=imu_gyro_noise,
            accel_noise=imu_accel_noise,
            gyro_rw=imu_gyro_bias_noise,
            accel_rw=imu_accel_bias_noise,
        )
        imu_preintegrations[(sw_ids[k], sw_ids[k + 1])] = preint

    sw_state = SlidingWindowState(window_size=N)

    result = initialize_visual_inertial_state(
        view_set=view_set,
        sliding_window=sw_state,
        imu_preintegrations=imu_preintegrations,
        view_ids=sw_ids,
        sensor_transform=(T_BS_R, T_BS_t),
        apply_scale_to_map=False,
    )

    print("\n========== Python VI Alignment ==========")
    print("success:   ", result.success)
    print("scale:     ", result.scale, "   (MATLAB:", matlab_scale, ")")
    print("gravity:   ", result.gravity)
    print("accel_bias:", result.accel_bias)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "vi_alignment_debug.mat")