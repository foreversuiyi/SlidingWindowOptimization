function [first_state, initial_states] = InitGuess(groundtruth, start_idx, window_size, all_imu_measure, time_stamp)

%% Initial states calculation using IMU measurement
initial_states = {};
first_state.rot = AxisAng2Rot(groundtruth.ang_axis(:,start_idx));
if isnan(first_state.rot(1,1))
    first_state.rot = eye(3);
end
first_state.pos = groundtruth.pos(:,start_idx);
first_state.index = start_idx;
initial_states{1} = first_state;
for idx = 1:window_size
    k = idx + start_idx;
    imu_measure.omega = all_imu_measure.omega(:, k-1);
    imu_measure.vel = all_imu_measure.vel(:, k-1);
    ts = time_stamp(k) - time_stamp(k-1);
    initial_states{idx+1} = imuMotionUpdate(initial_states{idx}, imu_measure, ts);
end