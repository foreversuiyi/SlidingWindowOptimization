function [update_states, lm_est] = UpdateStates(current_states, obs_lm_ids, lm_est, dx)

%% Update states by applying a step delta_x
update_states = current_states;
num_states = length(current_states);
for idx = 1:num_states-1
    delta_pos = dx(1+(idx-1)*6:3+(idx-1)*6);
    delta_ang = dx(4+(idx-1)*6:6+(idx-1)*6);    
    update_states{idx+1}.pos = current_states{idx+1}.pos + delta_pos;
    update_states{idx+1}.rot = AxisAng2Rot(delta_ang)*current_states{idx+1}.rot;
end

init_idx = (num_states-1)*6 + 1;
for i = 1:length(obs_lm_ids)
    idx = init_idx + (i-1)*3;
    delta_pos = dx(idx:idx+2);
    lm_est(:, obs_lm_ids(i)) = lm_est(:, obs_lm_ids(i)) + delta_pos;
end