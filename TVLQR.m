
clc; close all;

%% --- Simulation setup ---
Ts   = 0.01;             % sample time [s]
Tend = 600;              % total time [s]
N    = round(Tend/Ts);
t    = (0:N)*Ts;

% Nonlinear RTP plant params (same as before)
p.Ac = 0.0128;
p.Ar = 1.81e-11;
p.B  = 311;
p.C  = 5.26;
Tamb = 300;

% Drift on/off 
drift.enable = true;
drift.Ac_fun = @(k) p.Ac * (1 + 0.25*sin(2*pi*(k/N)*2));
drift.Ar_fun = @(k) p.Ar * (1 + 0.40*cos(2*pi*(k/N)*1.2));
drift.B_fun  = @(k) p.B  * (1 - 0.20*sin(2*pi*(k/N)*0.8));
drift.C_fun  = @(k) p.C  * (1 + 0.10*cos(2*pi*(k/N)*0.5));

% Measurement noise
sigma_T = 0.25;

% Input limits
u_min = 0;  u_max = 1.0;

%% --- Ramp-and-soak reference profile Tref(k) ---
T_ramp_start = Tamb;
T_ramp_top   = 1100;
T_soak       = 1100;
T_cool_end   = 800;

t_ramp_up  = 60;
t_soak     = 180;
t_cool     = 90;

n_ramp_up  = round(t_ramp_up/Ts);
n_soak     = round(t_soak/Ts);
n_cool     = round(t_cool/Ts);

Tref = zeros(1,N+1);
% ramp up
Tref(1:n_ramp_up+1) = linspace(T_ramp_start, T_ramp_top, n_ramp_up+1);
% soak
idx_soak_start = n_ramp_up+1;
idx_soak_end   = idx_soak_start + n_soak;
Tref(idx_soak_start:idx_soak_end) = T_soak;
% cool down
idx_cool_start = idx_soak_end;
idx_cool_end   = idx_cool_start + n_cool;
idx_cool_end   = min(idx_cool_end, N+1);
Tref(idx_cool_start:idx_cool_end) = linspace(T_soak, T_cool_end, ...
                                              idx_cool_end-idx_cool_start+1);
% hold final
if idx_cool_end < N+1
    Tref(idx_cool_end+1:end) = T_cool_end;
end

%% Sinusoidal reference instead of ramp-and-soak
% T_mean = 900;        % K
% T_amp  = 150;        % K amplitude
% T_period = 60;       % seconds
% omega = 2*pi/T_period;
% 
% Tref = T_mean + T_amp * sin(omega * t);

%% --- RLS init (same discrete model as your ID demo) ---
theta  = [0.99; 0.01; 0.0];   % [alpha; beta; gamma]
P      = 1e3*eye(3);
lambda = 0.999;
delta_huber = 3*sigma_T;
huber_w = @(e) (abs(e) <= delta_huber).*1.0 + ...
               (abs(e)  > delta_huber).*(delta_huber./abs(e));

% logs
T    = zeros(1,N+1);    T(1) = Tamb;
u    = zeros(1,N+1);
Tr   = Tref;
alpha_hat = zeros(1,N+1);
beta_hat  = zeros(1,N+1);
gamma_hat = zeros(1,N+1);
eps_log   = zeros(1,N);

% LQR weighting (on error and delta-u), design space
Q  = 1.0;         % penalize temperature tracking error
R  = 1e-4;        % penalize control increments (tune these)
Qf = Q;           % terminal cost


% Parameter and error logs for diagnostics
a_true = zeros(1,N+1);
b_true = zeros(1,N+1);
a_hat  = zeros(1,N+1);
b_hat  = zeros(1,N+1);

K_log  = zeros(1,N+1);   % LQR gains
u_ref_log = zeros(1,N+1);
u_fb_log  = zeros(1,N+1);

trace_P_log = zeros(1,N+1);
cond_P_log = zeros(1,N+1);
eig_P_log = zeros(3,N+1);
T_nom = 1;  % Kelvin
u_nom = 1;   
%% --- Main loop ---
for k = 1:N

    % --- Plant drift ---
    if drift.enable
        Ac = drift.Ac_fun(k);
        Ar = drift.Ar_fun(k);
        Bp = drift.B_fun(k);
        Cp = drift.C_fun(k);
    else
        Ac = p.Ac; Ar = p.Ar; Bp = p.B; Cp = p.C;
    end

    % --- Nonlinear plant step ---
    dTdt = -Ac*T(k) - Ar*T(k)^4 + Bp*u(k) + Cp;
    T_true_next = T(k) + Ts*dTdt;
    T_meas_next = T_true_next + sigma_T*randn;

    % --- RLS update (same as your robust RLS) ---
    T_scaled = T(k) / T_nom;
    phi = [T_scaled; u(k); 1];
    y   = T_meas_next;

    eps_k   = y - (phi.' * theta);
    eps_log(k) = eps_k;

    w = huber_w(eps_k);
    rt = sqrt(w);
    phi_t = rt * phi;
    y_t   = rt * y;
    eps_t = rt * (y - phi.'*theta);

    denom = lambda + (phi_t.' * P * phi_t);
    P = (P - (P * (phi_t*phi_t.') * P) / denom) / lambda;
    P = 0.5*(P + P.');
    trace_P_log(k+1) = trace(P);
    cond_P_log(k+1) = cond(P);
    eig_P_log(:,k+1) = eig(P);
    theta = theta + (P * phi_t) * eps_t;

    % Extract scaled parameters
    alpha_scaled = theta(1);  % Dimensionless
    beta_scaled  = theta(2);  % Dimensionless  
    gamma_scaled = theta(3);  % Dimensionless

    alpha_hat(k+1) = alpha_scaled;  % α acts on scaled T
    beta_hat(k+1)  = beta_scaled * T_nom;  % β needs rescaling
    gamma_hat(k+1) = gamma_scaled * T_nom;
%     alpha_hat(k+1) = theta(1);
%     beta_hat(k+1)  = theta(2);
%     gamma_hat(k+1) = theta(3);
    % True local linearization for plotting:
    a_true(k) = -Ac - 4*Ar*T(k)^3;   % continuous-time true a
    b_true(k) = Bp;                  % true b
    % Convert estimated discrete params to continuous-time approx
    a_hat(k) = (alpha_hat(k+1) - 1)/Ts;
    b_hat(k) =  beta_hat(k+1)/Ts;

    % --- Adaptive LQR design based on current alpha,beta ---
    A_d = alpha_hat(k+1);
    B_d = beta_hat(k+1);

    % If B is too small or sign-flips, guard it:
    if abs(B_d) < 1e-4
        B_d = sign(B_d + (B_d==0))*1e-4;
    end

    % For simplicity, solve TI-LQR on scalar system e_{k+1}=A_d e_k + B_d u_tilde
    % by iterating Riccati for a short finite horizon (e.g. Nh = 200 samples):
    Nh = 200;
    Pk = Qf;
    for j = 1:Nh
        % scalar TV/constant recursion
        S   = R + B_d'*Pk*B_d;
        K_l = (B_d'*Pk*A_d)/S;
        Pk  = Q + A_d'*Pk*A_d - A_d'*Pk*B_d*K_l;
    end
    % After backward pass, K_l is the first-step gain
    K = K_l;  % LQR gain for error
    % Log LQR gain
    K_log(k) = K;

    %  Reference feedforward u_ref(k) from affine model, optional
    if k < N
        Tr_k   = Tr(k);
        Tr_kp1 = Tr(k+1);
    else
        Tr_k   = Tr(k);
        Tr_kp1 = Tr(k);    % last sample
    end

    d_d = gamma_hat(k+1);
    u_ref_k = (Tr_kp1 - A_d*Tr_k - d_d) / B_d;
    

    % Optionally clamp feedforward
    u_ref_k = min(max(u_ref_k, u_min), u_max);

    % --- Apply feedback on error ---
    e_k = T(k) - Tr_k;
    u_tot = u_ref_k - K * e_k;
    %u_tot = - K * e_k;
    % Clamp actual input into [0,1]
    u(k+1) = min(max(u_tot, u_min), u_max);
    %u(k+1) = u_tot;
    % Log components of control
    u_ref_log(k) = u_ref_k;      % feedforward term
    u_fb_log(k)  = -K * e_k;     % feedback correction

    % --- Save next temperature ---
    T(k+1) = T_true_next;
end
a_true(end) = -Ac - 4*Ar*T(end)^3;
b_true(end) = Bp;

a_hat(end) = (alpha_hat(end) - 1)/Ts;
b_hat(end) =  beta_hat(end)/Ts;

figure;
subplot(2,1,1);
semilogy(t, trace_P_log, 'LineWidth', 1.4);
xlabel('Time [s]'); ylabel('trace(P)');
grid on; title('Covariance Size');

subplot(2,1,2);
semilogy(t, cond_P_log, 'LineWidth', 1.4);
xlabel('Time [s]'); ylabel('cond(P)');
grid on; title('Condition Number');
figure;

subplot(3,1,1);
semilogy(t, eig_P_log(1,:), 'LineWidth', 1.4);
xlabel('Time [s]'); ylabel('\lambda_1');
grid on; title('Eigenvalue 1');

subplot(3,1,2);
semilogy(t, eig_P_log(2,:), 'LineWidth', 1.4);
xlabel('Time [s]'); ylabel('\lambda_2');
grid on; title('Eigenvalue 2');

subplot(3,1,3);
semilogy(t, eig_P_log(3,:), 'LineWidth', 1.4);
xlabel('Time [s]'); ylabel('\lambda_3');
grid on; title('Eigenvalue 3');


figure;
subplot(2,1,1);
plot(t, a_true, 'k', 'LineWidth',1.4); hold on;
plot(t, a_hat, 'r--', 'LineWidth',1.4);
xlabel('Time [s]'); ylabel('a [1/s]');
legend('True','Estimated');
title('Parameter a(t)');
grid on;

subplot(2,1,2);
plot(t, b_true, 'k', 'LineWidth',1.4); hold on;
plot(t, b_hat, 'r--', 'LineWidth',1.4);
xlabel('Time [s]'); ylabel('b [K/(s·u)]');
legend('True','Estimated');
title('Parameter b(t)');
grid on;
figure;
subplot(2,1,1);
plot(t, a_true - a_hat, 'b', 'LineWidth',1.4);
xlabel('Time [s]'); ylabel('a_{true} - a_{hat}');
grid on;
title('a Parameter Error');

subplot(2,1,2);
plot(t, b_true - b_hat, 'b', 'LineWidth',1.4);
xlabel('Time [s]'); ylabel('b_{true} - b_{hat}');
grid on;
title('b Parameter Error');
figure;
plot(t, K_log, 'LineWidth', 1.4);
xlabel('Time [s]'); ylabel('LQR gain K');
grid on;
title('Adaptive LQR Gain Evolution');
figure;
plot(t, u_ref_log, 'r--', 'LineWidth', 1.2); hold on;
plot(t, u_fb_log, 'b', 'LineWidth', 1.2);
plot(t, u, 'k', 'LineWidth', 1.6);
legend('Feedforward u_{ref}', 'Feedback u_{fb}', 'Total u', ...
       'Location','Best');
xlabel('Time [s]'); ylabel('Input');
grid on;
title('Control Signal Decomposition');


%% --- Simple plots ---
figure;
subplot(2,1,1);
plot(t, T, 'LineWidth',1.4); hold on;
plot(t, Tr, 'r--', 'LineWidth',1.4);
grid on; ylabel('Temperature [K]');
legend('T','T_{ref}','Location','Best');
title('Adaptive LQR RTP: Temperature tracking');

subplot(2,1,2);
plot(t, u, 'LineWidth',1.4); grid on;
xlabel('Time [s]'); ylabel('u');
title('Lamp input (with adaptive LQR)');


