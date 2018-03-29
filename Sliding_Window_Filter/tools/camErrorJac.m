function [ext_error, ext_jacxk, ext_jaclk] = camErrorJac(k_state, valid_obs_num, cam, lm_est, cam_obs)

%% Compute the error vector and jacobians
ext_error = zeros(2*valid_obs_num, 1);
ext_jacxk = zeros(2*valid_obs_num, 6);
ext_jaclk = zeros(2*valid_obs_num, 3);
for k = 1:valid_obs_num
    idx = 1 + 2*(k -1);
    obs = cam_obs(:, 1, k);
    lm_pos = lm_est(:,k);
    cam_error = monoError(obs, k_state, cam, lm_pos);
    [jacxk, jaclk] = CalJacXL(k_state, cam, lm_pos);
    ext_error(idx:idx+1,:) = cam_error;
    ext_jacxk(idx:idx+1,:) = jacxk;
    ext_jaclk(idx:idx+1,:) = jaclk;
end

function mono_err = monoError(obs, k_state, cam, lm_pos)

%% Compute the error vector with a landmark
v_R_o = k_state.rot;
o_P_v = k_state.pos;
c_R_v = cam.c_R_v;
v_P_c = cam.v_P_c;
lm_pos_c = c_R_v*(v_R_o*(lm_pos - o_P_v) - v_P_c);

x = lm_pos_c(1);
y = lm_pos_c(2);
z = lm_pos_c(3);

cx = cam.cx;
cy = cam.cy;
fx = cam.fx;
fy = cam.fy;

cam_sim = [fx*x/z; fy*y/z] + [cx; cy];
mono_err = obs - cam_sim;
end

function [jacxk, jaclk] = CalJacXL(k_state, cam, lm_pos)

%% Compute the jacobian between camera state and landmark position
v_R_o = k_state.rot;
o_P_v = k_state.pos;
c_R_v = cam.c_R_v;
v_P_c = cam.v_P_c;
lm_pos_c = c_R_v*(v_R_o*(lm_pos - o_P_v) - v_P_c);

L = [-c_R_v*v_R_o, c_R_v*Vec2Skew(v_R_o*(lm_pos - o_P_v))];

x = lm_pos_c(1);
y = lm_pos_c(2);
z = lm_pos_c(3);

fx = cam.fx;
fy = cam.fy;

Jac = [fx/z, 0, -(1/z^2)*(fx*x); ...
       0, fy/z, -(1/z^2)*(fy*y)];

jacxk = Jac*L;
jaclk = Jac*c_R_v*v_R_o;
end
end