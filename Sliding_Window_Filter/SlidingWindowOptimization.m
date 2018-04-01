clc
clear
close all
addpath('tools')
file_name = '2011_09_26_drive_0001_sync_KLT.mat';
load(file_name);
%% Parameters Initialization
% Camera Parameters
cam.cx = cx;
cam.cy = cy;
cam.fx = fx;
cam.fy = fy;
cam.c_R_v = c_R_v;
cam.v_P_c = v_P_c;
cam.c_T_v = [c_R_v, -c_R_v*v_P_c; 0, 0, 0, 1];
cam.cov = 25*ones(2,1);
cam.x_cov = cam.cov(1)/cam.fx^2;
cam.y_cov = cam.cov(2)/cam.fy^2;
cam.K = [cam.fx, 0, cam.cx; 0, cam.fy, cam.cy; 0, 0, 1];
cam.invK = inv(cam.K);
% IMU Parameters
imu.vel_cov = 0.1*ones(3,1);
imu.omega_cov = 0.1*ones(3,1);
imu.cov = diag([imu.vel_cov; imu.omega_cov]);
% Sliding Window Parameters
num_lm = size(all_cam_obs, 3);
opt_lam = 1e-4;
update_lam = 0.25;
cost_thresh = 0.1e-2;
win_size = 10;
max_iter = 5;
start_idx = 1;
end_idx = size(all_cam_obs, 2) - win_size -1;

%% Create Initial Guess
[first_state, initial_states] = InitGuess(groundtruth, start_idx, win_size, all_imu_measure, time_stamp);

%% Calculate States Using IMU Only
imuonly_states = IMUOnlyStates(first_state, start_idx, end_idx, all_imu_measure, time_stamp);

%% Sliding Window Optimization Fusing the Camera and IMU
% Sliding Window
sliding_states{1} = first_state;
for win_idx = start_idx:end_idx
    win_end = win_idx + win_size;
    if win_idx == start_idx
        current_states = initial_states;
    else
        current_states = current_states(2:end);
        imu_measure.omega = all_imu_measure.omega(:,win_end-1);
        imu_measure.vel = all_imu_measure.vel(:,win_end-1);
        ts = time_stamp(win_end) - time_stamp(win_end-1);
        current_states{end+1} = imuMotionUpdate(current_states{end}, imu_measure, ts);
    end
    
    % Initialize Landmarks
    lm_est = zeros(3, num_lm);
    lm_obs = {};
    for lm_idx = 1:num_lm
        lm_obs{lm_idx}.cam_states = {};
        lm_obs{lm_idx}.cam_obs = [];
    end
    % Get the Observations in the Window
    for kidx = 1:win_size
        k = kidx + win_idx;
        valid_obs_idx = find(all_cam_obs(1, k, :) > -1);
        kstate = current_states{kidx+1};
        v_T_o = [kstate.rot -kstate.rot*kstate.pos; 0 0 0 1];
        c_T_o = cam.c_T_v*v_T_o;
        o_T_c = [c_T_o(1:3,1:3)', -c_T_o(1:3,1:3)'*c_T_o(1:3,4); 0 0 0 1];
        cam_state.rot = c_T_o(1:3,1:3);
        cam_state.pos = o_T_c(1:3,4);
        for obsidx = valid_obs_idx'
            lm_obs{obsidx}.cam_states{end+1} = cam_state;
            lm_obs{obsidx}.cam_obs(:,end+1) = [all_cam_obs(1:2,k,obsidx);1];
        end
    end
    
    % Calculate the Position of Landmarks using Triangulation
    total_lm_obs = 0;
    obs_lm_ids = [];
    unique_lm_obs = 0;
    for obslm_idx = 1:length(lm_obs)
        if length(lm_obs{obslm_idx}.cam_states) > 1
            cam_states = lm_obs{obslm_idx}.cam_states;
            cam_obs = lm_obs{obslm_idx}.cam_obs;
            % Triangulation and GN Optimization
            [lm_cal, cost] = GNPointEst(cam_states, cam_obs, cam.invK, [cam.x_cov; cam.y_cov]);
            if cost < cost_thresh * length(cam_states)^2    
                lm_est(:,obslm_idx) = lm_cal;
                total_lm_obs = total_lm_obs + length(lm_obs{obslm_idx}.cam_states);
                unique_lm_obs = unique_lm_obs + 1;
                obs_lm_ids(end+1) = obslm_idx;
            end
        end
    end
    % Start Iteration
    optimal_states = current_states;
    best_cost = Inf;
    delta_x = Inf;
    
    for iter_idx = 1:max_iter+1
        error = zeros(6*win_size+2*total_lm_obs, 1);
        error_idx = 1;
        Jacobian = sparse(6*win_size+2*total_lm_obs, 6*win_size+3*unique_lm_obs);
        Sigma = sparse(6*win_size+2*total_lm_obs, 6*win_size+2*total_lm_obs);
        Jacobian_idx = 1;
        Sigma_idx = 1;
        for kidx = 1:win_size
            k = kidx + win_idx;
            imu_measure.omega = all_imu_measure.omega(:,k-1);
            imu_measure.vel = all_imu_measure.vel(:,k-1);
            ts = time_stamp(k) - time_stamp(k-1);
            
            % Interoceptive IMU State Error and Jacobians
            k_state = current_states{kidx+1};
            pre_k_state = current_states{kidx};
            [int_error, int_jacxk, int_jacwk] = imuErrorJac(k_state, pre_k_state, imu_measure, ts);
            
            % Exteroceptive CAMERA Error and Jacobians
            valid_obs_idx = intersect(find(all_cam_obs(1,k,:) > -1), obs_lm_ids);
            if kidx == 1 && iter_idx == 1
                fprintf('Tracking %d landmarks. \n', length(valid_obs_idx));
            end
            if ~isempty(valid_obs_idx)
                [ext_error, ext_jacxk, ext_jaclk] = camErrorJac(k_state, length(valid_obs_idx), cam, lm_est(:,valid_obs_idx), all_cam_obs(1:2,k,valid_obs_idx));
            else
                ext_error = [];
            end
            
            % Error vector
            combined_error = [int_error; ext_error];
            error(error_idx:(error_idx+length(combined_error)-1),1) = combined_error;
            error_idx = error_idx + length(combined_error);
            % State Jacobian 
            jac = zeros(6 + 2*length(valid_obs_idx),12);
            jac(1:6,1:6) = -int_jacxk;
            jac(1:6,7:12) = eye(6);
            if ~isempty(valid_obs_idx)
                jac(7:(7+2*length(valid_obs_idx)-1),7:12) = -ext_jacxk;
            end
            num_jac_rows = size(jac, 1);
            Jacobian(Jacobian_idx:(Jacobian_idx + num_jac_rows -1), 1 + 6*(kidx-1):12+6*(kidx-1)) = jac;
            % Landmarks Position Jacobian
            lm_num = 1;
            for lm_id = valid_obs_idx'
                row_idx = 2*(lm_num-1)+1;
                col_lm_id = find(obs_lm_ids == lm_id);
                Jacobian(Jacobian_idx+6+row_idx-1:Jacobian_idx+6+row_idx, ...
                    6*(win_size+1)+3*col_lm_id-2:6*(win_size+1)+3*col_lm_id) = -ext_jaclk(row_idx:row_idx+1,:);
                lm_num = lm_num + 1;
            end
            Jacobian_idx = Jacobian_idx + num_jac_rows;
            
            % Covariance Matrix
            cov_k = zeros(6+2*length(valid_obs_idx), 6+2*length(valid_obs_idx));
            cov_k(1:6, 1:6) = int_jacwk*imu.cov*ts^2*int_jacwk';
            cov_k(7:end, 7:end) = diag(repmat(cam.cov, [length(valid_obs_idx),1]));
            num_covk_rows = size(cov_k, 1);
            Sigma(Sigma_idx:(Sigma_idx+num_covk_rows-1),Sigma_idx:(Sigma_idx+num_covk_rows-1)) = cov_k;
            Sigma_idx = Sigma_idx + num_covk_rows;
        end
        % Keep the first state unchanged
        Jacobian = Jacobian(:,7:end);
        new_cost = 0.5*error'*(Sigma\error);
        if new_cost < best_cost
            best_cost = new_cost;
            optimal_states = current_states;
        end
        
        if norm(delta_x) <1e-3
            break;
        end
        
        if iter_idx <= max_iter
            H = Jacobian'*(Sigma\Jacobian);
            H = H + opt_lam*diag(diag(H));
            b = Jacobian'*(Sigma\error);
            delta_x = -H\b;
            [current_states, lm_est] = UpdateStates(current_states, obs_lm_ids, lm_est, update_lam*delta_x);
        end
    end
    current_states = optimal_states;
    if iter_idx == max_iter
        disp('Warning: Failed to converge!');
    end

    fprintf('Window %d done, J = %.5f, %d iterations. \n', win_idx, best_cost, iter_idx);
    sliding_states{end+1} = current_states{2};
end

%% Ploting
g_x = zeros(end_idx-start_idx, 1); g_y = zeros(end_idx-start_idx, 1); g_z = zeros(end_idx-start_idx, 1);
i_x = zeros(end_idx-start_idx, 1); i_y = zeros(end_idx-start_idx, 1); i_z = zeros(end_idx-start_idx, 1);
s_x = zeros(end_idx-start_idx, 1); s_y = zeros(end_idx-start_idx, 1); i_z = zeros(end_idx-start_idx, 1);
for win_idx = start_idx:end_idx
    i_x(win_idx) = -imuonly_states{win_idx}.pos(2);
    i_y(win_idx) = imuonly_states{win_idx}.pos(1);
    i_z(win_idx) = imuonly_states{win_idx}.pos(3);
    g_x(win_idx) = -groundtruth.pos(2,win_idx);
    g_y(win_idx) = groundtruth.pos(1,win_idx);
    g_z(win_idx) = groundtruth.pos(3,win_idx);
    s_x(win_idx) = -sliding_states{win_idx}.pos(2);
    s_y(win_idx) = sliding_states{win_idx}.pos(1);
    s_z(win_idx) = sliding_states{win_idx}.pos(3);
end
figure
hold on
grid on
plot3(g_x,g_y,g_z,'k');
plot3(i_x,i_y,i_z,'r');
plot3(s_x,s_y,s_z,'g');
view(3)
legend('Groundtruth', 'IMU only', 'Sliding Window Filter')
rmpath('tools')