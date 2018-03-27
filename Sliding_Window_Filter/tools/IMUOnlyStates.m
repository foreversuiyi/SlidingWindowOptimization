function imuonly_states = IMUOnlyStates(first_state, start_idx, end_idx, all_imu_measure, time_stamp)

%% Calculate states using IMU only
imuonly_states{1} = first_state;
for idx = (start_idx+1):(end_idx+1)
    imu_measure.omega = all_imu_measure.omega(:, idx-1);
    imu_measure.vel = all_imu_measure.vel(:, idx-1);
    ts = time_stamp(idx) - time_stamp(idx-1);
    imuonly_states{idx} = imuMotionUpdate(imuonly_states{idx-1}, imu_measure, ts);
end