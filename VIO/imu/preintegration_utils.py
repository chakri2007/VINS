"""
preintegration_utils.py

Single shared entry point for "preintegrate the IMU samples between two
timestamps, linearized at a given bias" -- used by:

    vio_core.VisualInertialOdometry._build_imu_preintegrations
        (Phase 2 VI-alignment, consecutive keyframe pairs)

    vio_core.VisualInertialOdometry._build_single_imu_preintegration
        (Phase 3 BA_motion, previous keyframe -> not-yet-added frame)

    optimization.graph_builder.GraphBuilder.build_windowed_vio
        (Phase 3 windowed BA, consecutive keyframe pairs in the window)

Kept here (in imu/, which nothing else depends on) rather than in
vio_core.py specifically so optimization/graph_builder.py can import it
too without creating a vio_core.py <-> optimization circular import
(vio_core.py already imports GraphBuilder).
"""

from imu.preintegration import IMUPreintegrator
from memory_management.sliding_window import extract_imu_between


def preintegrate_between(
    sw_state,
    imu_calib,
    t_from,
    t_to,
    bias_g,
    bias_a,
):
    """
    Preintegrate every IMU sample in `sw_state.imu_buffer` between two
    timestamps, using (bias_g, bias_a) as the linearization point.

    Returns
    -------
    imu.preintegration.PreintegratedIMU, or None if there are fewer
    than 2 samples in [t_from, t_to] (insufficient IMU coverage).
    """

    samples = extract_imu_between(sw_state, t_from, t_to)

    if len(samples) < 2:
        return None

    preintegrator = IMUPreintegrator(
        gyro_noise=imu_calib.get(
            'gyroscope_noise_density', 1.0e-3
        ),
        accel_noise=imu_calib.get(
            'accelerometer_noise_density', 1.0e-2
        ),
        gyro_random_walk=imu_calib.get(
            'gyroscope_random_walk', 1.0e-5
        ),
        accel_random_walk=imu_calib.get(
            'accelerometer_random_walk', 1.0e-4
        ),
        bias_g=bias_g,
        bias_a=bias_a,
    )

    return preintegrator.integrate_measurements(samples)
