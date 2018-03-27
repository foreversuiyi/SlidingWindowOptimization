function new_state = imuMotionUpdate(old_state, imu_measurement, ts)
%% Update the state from IMU measurement after time ts

rot_vector = imu_measurement.omega*ts;
rot_matrix = AxisAng2Rot(rot_vector);
pos_vector = imu_measurement.vel*ts;

new_state.rot = rot_matrix*old_state.rot;
new_state.pos = old_state.pos + old_state.rot'*pos_vector;
new_state.index = old_state.index + 1;