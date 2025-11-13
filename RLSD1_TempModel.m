rtp_rls_robust_est_demo

function rtp_rls_robust_est_demo
clc; close all;

%% --- Simulation setup ---
Ts   = 0.01;               % sample time [s]
Tend = 600;                % total time [s]
N    = round(Tend/Ts);
t    = (0:N)*Ts;

% Nonlinear RTP plant params
p.Ac = 0.0128;             % [1/s]
p.Ar = 1.81e-11;           % [K^-3 / s]
p.B  = 311;                % [K/s] per unit input
p.C  = 5.26;               % [K/s]
Tamb = 300;                % ambient [K]

% Optional slow drift (toggle on/off)
drift.enable = true;
drift.Ac_fun = @(k) p.Ac * (1 + 0.25*sin(2*pi*(k/N)*2));       % ±25%
drift.Ar_fun = @(k) p.Ar * (1 + 0.40*cos(2*pi*(k/N)*1.2));     % ±40%
drift.B_fun  = @(k) p.B  * (1 - 0.20*sin(2*pi*(k/N)*0.8));     % ±20%
drift.C_fun  = @(k) p.C  * (1 + 0.10*cos(2*pi*(k/N)*0.5));     % ±10%

% Measurement noise
sigma_T = 0.25;            % [K] std dev

% Input limits
u_min = 0;  u_max = 1.0;   % normalized lamp command

%% --- Excitation input (PRBS + small dither) ---
blk_sec   = 2.0;                         % PRBS block length [s]
blk_len   = max(1, round(blk_sec/Ts));   % samples per block
dither_amp = 0.02;                       % small dither (helps PE)
rng(1);                                  % reproducible PRBS

prbs_level = 0.5;                        % start mid-level
prbs_hist  = prbs_level;

u = zeros(1,N+1);

%% --- RLS initialization (ARX with bias) ---
% Model: T_{k+1} = alpha*T_k + beta*u_k + gamma
theta  = [0.99; 0.01; 0.0];         % [alpha; beta; gamma] initial guess
P      = 1e3*eye(3);                % large initial covariance
lambda = 0.995;                     % forgetting factor (as requested)
b_guard = 1e-3;                     % avoids division by tiny b_hat (for plotting only)

% Robust M-estimator (Huber) for residuals
delta_huber = 3*sigma_T;           % threshold ~3σ
huber_w = @(e) (abs(e) <= delta_huber) .* 1.0 + ...
               (abs(e)  > delta_huber) .* (delta_huber ./ abs(e));

%% --- Allocate logs ---
T   = zeros(1,N+1);
alpha_hat = zeros(1,N+1); beta_hat = zeros(1,N+1); gamma_hat = zeros(1,N+1);
a_true = zeros(1,N+1);    b_true = zeros(1,N+1);
a_hat  = zeros(1,N+1);    b_hat  = zeros(1,N+1);
eps_log = zeros(1,N);

% Init condition
T(1)  = Tamb;

%% --- Main loop ---
for k = 1:N
    % --- Possibly drift plant params ---
    if drift.enable
        Ac = drift.Ac_fun(k);
        Ar = drift.Ar_fun(k);
        Bp = drift.B_fun(k);
        Cp = drift.C_fun(k);
    else
        Ac = p.Ac; Ar = p.Ar; Bp = p.B; Cp = p.C;
    end

    % --- True local linearization (for plotting only) ---
    a_true(k) = -Ac - 4*Ar*T(k)^3;
    b_true(k) =  Bp;

    % --- Excitation input (PRBS + dither) ---
    if mod(k-1, blk_len) == 0
        % toggle between two levels in [u_min, u_max]
        prbs_level = u_min + (u_max - u_min) * (rand > 0.5);
        prbs_hist  = prbs_level;
    else
        prbs_level = prbs_hist;
    end
    u_cmd = prbs_level + dither_amp*sin(2*pi*0.5*t(k+1));
    u(k)  = min(max(u_cmd, u_min), u_max);

    % --- Nonlinear plant step:  \dot T = -Ac T - Ar T^4 + B u + C  ---
    dTdt  = -Ac*T(k) - Ar*T(k)^4 + Bp*u(k) + Cp;
    T_true_next = T(k) + Ts*dTdt;

    % Measurement (noisy)
    T_meas_next = T_true_next + sigma_T*randn;

    % --- Convert theta to CT params (for plotting) ---
    alpha_hat(k) = theta(1);
    beta_hat(k)  = theta(2);
    gamma_hat(k) = theta(3);
    a_hat(k) = (alpha_hat(k) - 1)/Ts;
    b_hat(k) =  beta_hat(k)/Ts;

    % --- Robust DTRLS (P-first form; matches given equations) ---
    % Regressor and target
    phi = [T(k); u(k); 1];        % 3x1
    y   = T_meas_next;

    % Innovation with previous theta
    eps_k = y - (phi.' * theta);  % scalar
    eps_log(k) = eps_k;

    % Robust weighting (Huber)
    w = huber_w(eps_k);
    rt = sqrt(w);
    phi_t = rt * phi;             % weighted regressor
    y_t   = rt * y;               % weighted target
    eps_t = rt * (y - phi.'*theta);

    % Covariance update (P-first):
    % P_k = (P_{k-1} - P_{k-1}*phi_t*phi_t'*P_{k-1} / (lambda + phi_t'*P_{k-1}*phi_t)) / lambda
    denom = lambda + (phi_t.' * P * phi_t);
    P = (P - (P * (phi_t*phi_t.') * P) / denom) / lambda;

    % Symmetrize (numeric hygiene)
    P = 0.5 * (P + P.');

    % Parameter update:
    % theta_k = theta_{k-1} + P_k * phi_t * eps_t
    theta = theta + (P * phi_t) * eps_t;

    % --- Advance true state ---
    T(k+1) = T_true_next;
end

% last-step param logs
alpha_hat(end) = theta(1); beta_hat(end) = theta(2); gamma_hat(end) = theta(3);
a_hat(end) = (alpha_hat(end)-1)/Ts; b_hat(end) = beta_hat(end)/Ts;
a_true(end) = -Ac - 4*Ar*T(end)^3; b_true(end) = Bp;

%% --- Plots ---
figure; subplot(3,1,1);
plot(t,T,'LineWidth',1.4); grid on; ylabel('Temperature [K]');
title('RTP Temperature (drifting plant, no control)');

subplot(3,1,2);
plot(t(1:end-1),u(1:end-1),'LineWidth',1.4); grid on;
ylabel('u'); title('Excitation Input (PRBS + dither)'); ylim([u_min-0.05 u_max+0.05]);

subplot(3,1,3);
plot(t(1:end-1),eps_log,'LineWidth',1.0); grid on; xlabel('Time [s]'); ylabel('Residual \epsilon_k');
title('Measurement Innovation (for robustness inspection)');

figure; subplot(3,1,1);
plot(t,a_true,'k','LineWidth',1.2); hold on; plot(t,a_hat,'r--','LineWidth',1.2);
ylabel('a [1/s]'); legend('true','hat'); grid on; title('Parameter a(t)');

subplot(3,1,2);
plot(t,b_true,'k','LineWidth',1.2); hold on; plot(t,b_hat,'r--','LineWidth',1.2);
ylabel('b [K/(s·u)]'); legend('true','hat'); grid on; title('Parameter b(t)');

subplot(3,1,3);
plot(t,gamma_hat,'LineWidth',1.2); grid on; xlabel('Time [s]'); ylabel('\gamma');
title('Bias term \gamma (offset)');

% Separate figure for errors
figure;
subplot(2,1,1);
plot(t, a_true - a_hat, 'b', 'LineWidth', 1.2);
ylabel('a error [1/s]'); 
grid on; 
title('Parameter a(t) Error (true - hat)');
xlabel('Time [s]');

subplot(2,1,2);
plot(t, b_true - b_hat, 'b', 'LineWidth', 1.2);
ylabel('b error [K/(s·u)]'); 
grid on; 
title('Parameter b(t) Error (true - hat)');
xlabel('Time [s]');
fprintf('Final estimates: a_hat=%.4f  b_hat=%.2f  (true last: a=%.4f, b=%.2f)\n', ...
         a_hat(end), b_hat(end), a_true(end), b_true(end));
end
