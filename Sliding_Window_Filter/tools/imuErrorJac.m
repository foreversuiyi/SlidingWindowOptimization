function [int_error, jac_xk, jac_wk] = imuErrorJac(k_state, pre_k_state, imu_measure, ts)

%% Compute the 6x1 IMU error vector between measurement and estimation
imu_trans = imu_measure.vel*ts;
imu_axis_ang = imu_measure.omega*ts;
imu_rot = AxisAng2Rot(imu_axis_ang);

error_rot_matrix = k_state.rot*(imu_rot*pre_k_state.rot)';
error_rot = [error_rot_matrix(2,3); error_rot_matrix(3,1); error_rot_matrix(1,2)];
error_trans = k_state.pos - (pre_k_state.pos + pre_k_state.rot'*imu_trans);

int_error = [error_trans; error_rot];

%% Compute the 6x6 Jacobian jac_xk
jac_xk = zeros(6,6);
jac_xk(1:3,1:3) = eye(3);
jac_xk(4:6,4:6) = imu_rot;
jac_xk(1:3,4:6) = -pre_k_state.rot'*Vec2Skew(imu_trans);

%% Compute the 6x6 Jacobian jac_wk
jac_wk = zeros(6,6);
jac_wk(1:3,1:3) = pre_k_state.rot';
jac_wk(4:6,4:6) = eye(3);

end
