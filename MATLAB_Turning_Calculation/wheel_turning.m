clc;
clear;

% === Known parameters ===
H = 316.17;
L = 346.47;
l1 = 96.236;
l2 = 85.82;
l3 = 411.2;
w = 632.34;
d = 1408;
const = 0.49411;

% === Define angle range from -60 deg to +60 deg in radians ===
angle_range = -pi/3 : pi/90 : pi/3;  % from -60° to +60°

% === Preallocate arrays ===
num_points = length(angle_range);
A_list = zeros(1, num_points);
left_wheel_list = zeros(1, num_points);
right_wheel_list = zeros(1, num_points);
color_list = false(1, num_points);  % true = green, false = red

% === Loop through steering angles ===
idx = 1;
for A = angle_range
    % Define anonymous functions
    f_left = @(B_left) (H + l2*sin(B_left) - l1*sin(A - 0.1565))^2 + ...
                       (L - l2*cos(B_left) - l1*cos(A - 0.1565))^2 - l3^2;

    f_right = @(B_right) (H + l2*sin(B_right) + l1*sin(A + 0.1565))^2 + ...
                         (L - l1*cos(A + 0.1565) - l2*cos(B_right))^2 - l3^2;

    try
        % Solve for B_left and B_right
        B_sol_left = fzero(f_left, 0);
        B_sol_right = fzero(f_right, 0);

        % Calculate wheel angles (convert to degrees)
        left_wheel = (B_sol_left - const) * 180 / pi;
        right_wheel = (B_sol_right - const) * 180 / pi;

        % Check for turning point
        f_row = @(r) tan(B_sol_left - const)*(w + r) - tan(B_sol_right - const)*r;
        r_sol = fzero(f_row, 1);
        h = tan(B_sol_right);
        turning_exists = abs(d - h) < 1e-2;

        % Save results
        A_list(idx) = A * 180 / pi;
        left_wheel_list(idx) = -left_wheel;
        right_wheel_list(idx) = right_wheel;
        color_list(idx) = turning_exists;
    catch
        % If fzero fails, mark as NaN
        A_list(idx) = A * 180 / pi;
        left_wheel_list(idx) = NaN;
        right_wheel_list(idx) = NaN;
        color_list(idx) = false;
    end

    idx = idx + 1;
end

% === Plotting ===
figure;
hold on;
grid on;
xlabel('Steering Angle A (degrees)');
ylabel('Wheel Angle (degrees)');
title('Left & Right Wheel Steering Angles');

for i = 1:num_points
    if isnan(left_wheel_list(i)) || isnan(right_wheel_list(i))
        continue;  % Skip invalid calculations
    end
    if color_list(i)
        plot(A_list(i), left_wheel_list(i), 'gx');   % green 'x' for left
        plot(A_list(i), right_wheel_list(i), 'go');  % green 'o' for right
    else
        plot(A_list(i), left_wheel_list(i), 'rx');   % red 'x' for left
        plot(A_list(i), right_wheel_list(i), 'ro');  % red 'o' for right
    end
end

legend('Left Wheel (x)', 'Right Wheel (o)', 'Location', 'best');

clc;
clear;

% === Known parameters ===
H = 316.17;
L = 346.47;
l1 = 96.236;
l2 = 85.82;
l3 = 411.2;
w = 632.34;
d = 1408;
const = 0.49411;
%left_data_x = [-40,-35,-30,-25,-20,-15,-10,-5,0,5,10,15,20,25,30,35,40];
%left_data_y = [25,25,22,20,16,12,8,4,0,-4,-9,-15,-19,-25,-32,-38,-43];
left_data_x = [40,35,30,25,20,15,10,5,0,-5,-10,-15,-20,-25,-30,-35,-40];
left_data_y = [-25,-25,-22,-20,-16,-12,-8,-4,0,4,9,15,19,25,32,38,43];
% === Define angle range from -60 deg to +60 deg in radians ===
angle_range = -pi/3 : pi/90 : pi/3;  % from -60° to +60°

% === Preallocate arrays ===
num_points = length(angle_range);
A_list = zeros(1, num_points);
left_wheel_list = zeros(1, num_points);
right_wheel_list = zeros(1, num_points);
color_list = false(1, num_points);  % true = green, false = red

% === Loop through steering angles ===
idx = 1;
for A = angle_range
    % Define anonymous functions
    f_left = @(B_left) (H + l2*sin(B_left) - l1*sin(A - 0.1565))^2 + ...
                       (L - l2*cos(B_left) - l1*cos(A - 0.1565))^2 - l3^2;

    f_right = @(B_right) (H + l2*sin(B_right) + l1*sin(A + 0.1565))^2 + ...
                         (L - l1*cos(A + 0.1565) - l2*cos(B_right))^2 - l3^2;

    try
        % Solve for B_left and B_right
        B_sol_left = fzero(f_left, 0);
        B_sol_right = fzero(f_right, 0);

        % Calculate wheel angles (convert to degrees)
        left_wheel = (B_sol_left - const) * 180 / pi;
        right_wheel = (B_sol_right - const) * 180 / pi;

        % Check for turning point
        f_row = @(r) tan(B_sol_left - const)*(w + r) - tan(B_sol_right - const)*r;
        r_sol = fzero(f_row, 1);
        h = tan(B_sol_right);
        turning_exists = abs(d - h) < 1e-2;

        % Save results
        A_list(idx) = A * 180 / pi;
        left_wheel_list(idx) = -left_wheel;
        right_wheel_list(idx) = right_wheel;
        color_list(idx) = turning_exists;
    catch
        % If fzero fails, mark as NaN
        A_list(idx) = A * 180 / pi;
        left_wheel_list(idx) = NaN;
        right_wheel_list(idx) = NaN;
        color_list(idx) = false;
    end

    idx = idx + 1;
end

% === Plotting only Left Wheel ===
figure;
hold on;
grid on;
xlabel('Steering Angle A (degrees)');
ylabel('Wheel Angle (degrees)');
title('Left Wheel Steering Angle Theory vs Actual');

for i = 1:num_points
    if isnan(left_wheel_list(i))
        continue;  % Skip invalid values
    end
    if color_list(i)
        plot(A_list(i), left_wheel_list(i), 'gx');   % green 'x' for left
    else
        plot(A_list(i), left_wheel_list(i), 'rx');   % red 'x' for left
    end
end

h_actual = plot(left_data_x,left_data_y, 'bx-', 'LineWidth', 1, 'MarkerSize', 6);  % blue line with circles


%legend('Left Wheel (x)', 'Left Wheel actual (x)','Location', 'best');
h_sim = plot(NaN, NaN, 'rx');  % dummy red x for legend

legend([h_sim, h_actual], {'Left Wheel theory (x)', 'Left Wheel actual (x)'}, 'Location', 'best');


% === Plotting only Right Wheel ===
figure;
hold on;
grid on;
xlabel('Steering Angle A (degrees)');
ylabel('Wheel Angle (degrees)');
title('Right Wheel Steering Angle Theory vs Actual');
%right_data_x = [-40,-35,-30,-25,-20,-15,-10,-5,0,5,10,15,20,25,30,35,40];
%right_data_y = [44,38,32,26,21,16,10,5,0,-5,-10,-15,-19,-22,-25,-26,-27];
right_data_x = [40,35,30,25,20,15,10,5,0,-5,-10,-15,-20,-25,-30,-35,-40];
right_data_y = [-44,-38,-32,-26,-21,-16,-10,-5,0,5,10,15,19,22,25,26,27];
for i = 1:num_points
    if isnan(right_wheel_list(i))
        continue;  % Skip invalid calculations
    end
    if color_list(i)
        plot(A_list(i), right_wheel_list(i), 'go');  % green 'o' for right
    else
        plot(A_list(i), right_wheel_list(i), 'ro');  % red 'o' for right
    end
end
h_actual = plot(right_data_x,right_data_y, 'bo-', 'LineWidth', 1, 'MarkerSize', 6);  % blue line with circles

%legend('Right Wheel (o)', 'right Wheel actual (o)','Location', 'best');
h_sim = plot(NaN, NaN, 'ro');  % dummy red o for legend

legend([h_sim, h_actual], {'Right Wheel theory(o)', 'Right Wheel actual (o)'}, 'Location', 'best');





clc; clear; close all;

%% === Known parameters ===
H = 316.17;
L = 346.47;
l1 = 96.236;
l2 = 85.82;
l3 = 411.2;
w  = 632.34;
d  = 1408;
const = 0.49411;      % rad, offset from your model

% If you had camber/roll you could include them, but we ignore here:
% epsilon_camber = 0;  % not used in conversion below

%% === Caster angle (steering axis tilt, side view) ===
eps_deg = 19.4;                 % <-- as requested
eps_rad = deg2rad(eps_deg);

%% === Left wheel actual data (A in deg, Left motor angle in deg) ===
left_data_x = [40,35,30,25,20,15,10,5,0,-5,-10,-15,-20,-25,-30,-35,-40];
left_data_y = [-25,-25,-22,-20,-16,-12,-8,-4,0,4,9,15,19,25,32,38,43];

%% === Define A range from -60 deg to +60 deg (in radians) ===
angle_range = -pi/3 : pi/90 : pi/3;  % from -60° to +60°
num_points  = length(angle_range);

%% === Preallocate arrays ===
A_list            = zeros(1, num_points);  % deg
left_wheel_list   = nan(1, num_points);    % deg (MOTOR angle after caster conversion)
right_wheel_list  = nan(1, num_points);    % deg (not plotted but computed)
color_list        = false(1, num_points);  % true = green, false = red (turning_exists)

%% === Loop through steering angles ===
idx = 1;
for A = angle_range
    % Geometry equations (solve B_left, B_right in radians)
    f_left = @(B_left)  (H + l2*sin(B_left) - l1*sin(A - 0.1565))^2 + ...
                        (L - l2*cos(B_left) - l1*cos(A - 0.1565))^2 - l3^2;

    f_right = @(B_right) (H + l2*sin(B_right) + l1*sin(A + 0.1565))^2 + ...
                         (L - l1*cos(A + 0.1565) - l2*cos(B_right))^2 - l3^2;

    try
        % Solve for left/right kingpin angles (radians)
        B_sol_left  = fzero(f_left,  0);
        B_sol_right = fzero(f_right, 0);

        % Horizontal-plane steering angles (δ_temp) in radians
        delta_temp_left  = B_sol_left  - const;
        delta_temp_right = B_sol_right - const;

        % === Convert δ_temp -> δ_motor with caster (φ = 0):
        % δ_motor = atan( tan(δ_temp) * cos(eps) )
        left_motor_deg  = rad2deg( atan( tan(delta_temp_left)  * cos(eps_rad) ) );
        right_motor_deg = rad2deg( atan( tan(delta_temp_right) * cos(eps_rad) ) );

        % Turning point check (same form you had; units aren't critical for the flag)
        f_row = @(r) tan(delta_temp_left)*(w + r) - tan(delta_temp_right)*r;
        r_sol = fzero(f_row, 1);
        h     = tan(B_sol_right);
        turning_exists = abs(d - h) < 1e-2;

        % Save results (note your original code used a minus sign on left)
        A_list(idx)           = rad2deg(A);
        left_wheel_list(idx)  = -left_motor_deg;    % keep your sign convention
        right_wheel_list(idx) =  right_motor_deg;
        color_list(idx)       = turning_exists;

    catch
        % If fzero fails, keep NaN and mark as not turning
        A_list(idx)           = rad2deg(A);
        left_wheel_list(idx)  = NaN;
        right_wheel_list(idx) = NaN;
        color_list(idx)       = false;
    end

    idx = idx + 1;
end

%% === Plotting only Left Wheel ===
figure; hold on; grid on;
xlabel('Steering Angle A (degrees)');
ylabel('Left Motor Angle (degrees)');
title('Left Wheel: Theory (with caster 19.4°) vs Actual');

for i = 1:num_points
    if isnan(left_wheel_list(i)), continue; end
    if color_list(i)
        plot(A_list(i), left_wheel_list(i), 'gx');   % green 'x' for left (turning_exists)
    else
        plot(A_list(i), left_wheel_list(i), 'rx');   % red 'x' for left
    end
end

h_actual = plot(left_data_x, left_data_y, 'bo-', 'LineWidth', 1, 'MarkerSize', 6);  % blue line with circles
h_sim    = plot(NaN, NaN, 'rx');  % dummy red x for legend

legend([h_sim, h_actual], {'Left Wheel theory (x)', 'Left Wheel actual (o)'}, 'Location', 'best');



%% === Plotting only Right Wheel ===
figure; hold on; grid on;
xlabel('Steering Angle A (degrees)');
ylabel('Right Motor Angle (degrees)');
title('Right Wheel: Theory (with caster 19.4°) vs Actual');

% Right wheel actual data
right_data_x = [40,35,30,25,20,15,10,5,0,-5,-10,-15,-20,-25,-30,-35,-40];
right_data_y = [-44,-38,-32,-26,-21,-16,-10,-5,0,5,10,15,19,22,25,26,27];

% Loop through and plot calculated right wheel angles
for i = 1:num_points
    if isnan(right_wheel_list(i))
        continue;  % Skip invalid calculations
    end
    if color_list(i)
        plot(A_list(i), right_wheel_list(i), 'go');  % green 'o' for turning_exists
    else
        plot(A_list(i), right_wheel_list(i), 'ro');  % red 'o' otherwise
    end
end

% Plot actual right wheel data (blue line with circles)
h_actual = plot(right_data_x, right_data_y, 'bo-', 'LineWidth', 1, 'MarkerSize', 6);

% Dummy red 'o' for legend
h_sim = plot(NaN, NaN, 'ro');

% Legend
legend([h_sim, h_actual], ...
    {'Right Wheel theory (o)', 'Right Wheel actual (o)'}, 'Location', 'best');




clc; clear; close all;

%% === Known parameters ===
H = 316.17;
L = 346.47;
l1 = 96.236;
l2 = 85.82;
l3 = 411.2;
w  = 632.34;
d  = 1408;
const = 0.49411;      % rad, offset from your model

%% === Caster angle (steering axis tilt, side view) ===
eps_deg = 19.4;                 % requested
eps_rad = deg2rad(eps_deg);

%% === Left wheel actual data (A in deg, Left motor angle in deg) ===
left_data_x = [40,35,30,25,20,15,10,5,0,-5,-10,-15,-20,-25,-30,-35,-40];
left_data_y = [-25,-25,-22,-20,-16,-12,-8,-4,0,4,9,15,19,25,32,38,43];

%% === Define A range from -60 deg to +60 deg (in radians) ===
angle_range = -pi/3 : pi/90 : pi/3;  % from -60° to +60°
num_points  = length(angle_range);

%% === Preallocate arrays ===
A_list            = zeros(1, num_points);  % deg
left_wheel_list   = nan(1, num_points);    % deg (MOTOR angle after caster conversion)
right_wheel_list  = nan(1, num_points);    % deg (not plotted but computed)
color_list        = false(1, num_points);  % true = green, false = red (turning_exists)

%% === Loop through steering angles ===
idx = 1;
for A = angle_range
    % Geometry equations (solve B_left, B_right in radians)
    f_left = @(B_left)  (H + l2*sin(B_left) - l1*sin(A - 0.1565))^2 + ...
                        (L - l2*cos(B_left) - l1*cos(A - 0.1565))^2 - l3^2;

    f_right = @(B_right) (H + l2*sin(B_right) + l1*sin(A + 0.1565))^2 + ...
                         (L - l1*cos(A + 0.1565) - l2*cos(B_right))^2 - l3^2;

    try
        % Solve for left/right kingpin angles (radians)
        B_sol_left  = fzero(f_left,  0);
        B_sol_right = fzero(f_right, 0);

        % Horizontal-plane steering angles (δ_temp) in radians
        delta_temp_left  = B_sol_left  - const;
        delta_temp_right = B_sol_right - const;

        % Convert δ_temp -> δ_motor with caster (φ = 0):
        % δ_motor = atan( tan(δ_temp) * cos(eps) )
        left_motor_deg  = rad2deg( atan( tan(delta_temp_left)  * cos(eps_rad) ) );
        right_motor_deg = rad2deg( atan( tan(delta_temp_right) * cos(eps_rad) ) );

        % Turning point check (flag only)
        f_row = @(r) tan(delta_temp_left)*(w + r) - tan(delta_temp_right)*r;
        r_sol = fzero(f_row, 1); %#ok<NASGU>
        h     = tan(B_sol_right);
        turning_exists = abs(d - h) < 1e-2;

        % Save results (keep your left sign convention)
        A_list(idx)           = rad2deg(A);
        left_wheel_list(idx)  = -left_motor_deg;    % theory motor angle (LEFT)
        right_wheel_list(idx) =  right_motor_deg;
        color_list(idx)       = turning_exists;

    catch
        A_list(idx)           = rad2deg(A);
        left_wheel_list(idx)  = NaN;
        right_wheel_list(idx) = NaN;
        color_list(idx)       = false;
    end

    idx = idx + 1;
end

%% === Prepare data in [-25, 25] for fitting/plotting ===
domain_min = -25; domain_max = 25;

% Clean theory (remove NaNs), then mask domain
valid_theory = ~isnan(left_wheel_list);
A_theory = A_list(valid_theory);
Y_theory = left_wheel_list(valid_theory);

mask_theory = (A_theory >= domain_min) & (A_theory <= domain_max);
A_theory_d = A_theory(mask_theory);          % deg
Y_theory_d = Y_theory(mask_theory);          % deg (motor)

% Actual data masked to domain
mask_actual = (left_data_x >= domain_min) & (left_data_x <= domain_max);
A_actual_d = left_data_x(mask_actual);
Y_actual_d = left_data_y(mask_actual);

%% === Fit cubic (order 3) to theory & actual (within domain) ===
order = 3;
p_theory = polyfit(A_theory_d, Y_theory_d, order);
p_actual = polyfit(A_actual_d, Y_actual_d, order);

% Pretty-print formulas
syms A_sym;
theory_expr = poly2sym(p_theory, A_sym);
actual_expr = poly2sym(p_actual, A_sym);

fprintf('\nCubic Fit (Theory motor, LEFT) over [-25,25]:\n  δ_theory(A) =\n'); disp(theory_expr);
fprintf('Cubic Fit (Actual, LEFT) over [-25,25]:\n  δ_actual(A) =\n');     disp(actual_expr);

%% === Evaluate errors on A = -25:2:25 ===
A_eval = domain_min:2:domain_max;
theory_fit_eval = polyval(p_theory, A_eval);
actual_fit_eval = polyval(p_actual, A_eval);

err = actual_fit_eval - theory_fit_eval;          % actual - theory (deg)
MAE  = mean(abs(err));
RMSE = sqrt(mean(err.^2));
MaxE = max(abs(err));
MPE  = mean( abs(err) ./ max(1e-9, abs(theory_fit_eval)) * 100 );  % mean percent error

fprintf('\nErrors on A = -25:2:25 (deg):\n');
fprintf('  MAE  = %.3f deg\n', MAE);
fprintf('  RMSE = %.3f deg\n', RMSE);
fprintf('  MaxE = %.3f deg\n', MaxE);
fprintf('  Mean Percent Error = %.3f %%\n', MPE);

%% === Plot (domain: [-25,25]) ===
figure; hold on; grid on;
xlabel('Steering Angle A (degrees)');
ylabel('Left Motor Angle (degrees)');
title('Left Wheel (Motor Angle): Theory vs Actual — cubic fits on [-25°,25°]');

% Raw points within domain
plot(A_theory_d, Y_theory_d, 'rx', 'DisplayName','Theory (raw, motor)');
plot(A_actual_d, Y_actual_d, 'bo', 'DisplayName','Actual (raw)');

% Fitted curves over dense grid in domain
A_plot = linspace(domain_min, domain_max, 800);
plot(A_plot, polyval(p_theory, A_plot), 'r-',  'LineWidth', 1., 'DisplayName','Theory cubic fit');
plot(A_plot, polyval(p_actual, A_plot), 'b--', 'LineWidth', 1., 'DisplayName','Actual cubic fit');

xlim([domain_min, domain_max]);
legend('Location','best');



right_data_x = [40,35,30,25,20,15,10,5,0,-5,-10,-15,-20,-25,-30,-35,-40];
right_data_y = [-44,-38,-32,-26,-21,-16,-10,-5,0,5,10,15,19,22,25,26,27];
%% === Fit and Compare Right Wheel in [-25°,25°] ===
domain_min = -25; domain_max = 25;

% Clean theory (remove NaNs), then mask domain
valid_theory_R = ~isnan(right_wheel_list);
A_theory_R = A_list(valid_theory_R);
Y_theory_R = right_wheel_list(valid_theory_R);

mask_theory_R = (A_theory_R >= domain_min) & (A_theory_R <= domain_max);
A_theory_Rd = A_theory_R(mask_theory_R);
Y_theory_Rd = Y_theory_R(mask_theory_R);

% Actual right data masked to domain
mask_actual_R = (right_data_x >= domain_min) & (right_data_x <= domain_max);
A_actual_Rd = right_data_x(mask_actual_R);
Y_actual_Rd = right_data_y(mask_actual_R);

% Fit cubic polynomials
order = 3;
p_theory_R = polyfit(A_theory_Rd, Y_theory_Rd, order);
p_actual_R = polyfit(A_actual_Rd, Y_actual_Rd, order);

% Pretty-print formulas
syms A_sym;
theory_expr_R = poly2sym(p_theory_R, A_sym);
actual_expr_R = poly2sym(p_actual_R, A_sym);

fprintf('\nCubic Fit (Theory motor, RIGHT) over [-25,25]:\n  δ_theory(A) =\n'); disp(theory_expr_R);
fprintf('Cubic Fit (Actual, RIGHT) over [-25,25]:\n  δ_actual(A) =\n');     disp(actual_expr_R);

% Evaluate errors on A = -25:2:25
A_eval_R = domain_min:2:domain_max;
theory_fit_eval_R = polyval(p_theory_R, A_eval_R);
actual_fit_eval_R = polyval(p_actual_R, A_eval_R);

err_R = actual_fit_eval_R - theory_fit_eval_R;
MAE_R  = mean(abs(err_R));
RMSE_R = sqrt(mean(err_R.^2));
MaxE_R = max(abs(err_R));
MPE_R  = mean( abs(err_R) ./ max(1e-9, abs(theory_fit_eval_R)) * 100 );

fprintf('\nRight Wheel Errors on A = -25:2:25 (deg):\n');
fprintf('  MAE  = %.3f deg\n', MAE_R);
fprintf('  RMSE = %.3f deg\n', RMSE_R);
fprintf('  MaxE = %.3f deg\n', MaxE_R);
fprintf('  Mean Percent Error = %.3f %%\n', MPE_R);

% Plot
figure; hold on; grid on;
xlabel('Steering Angle A (degrees)');
ylabel('Right Motor Angle (degrees)');
title('Right Wheel (Motor Angle): Theory vs Actual — cubic fits on [-25°,25°]');

% Raw points
plot(A_theory_Rd, Y_theory_Rd, 'ro', 'DisplayName','Theory (raw, motor)');
plot(A_actual_Rd, Y_actual_Rd, 'bo', 'DisplayName','Actual (raw)');

% Fitted curves
A_plot_R = linspace(domain_min, domain_max, 800);
plot(A_plot_R, polyval(p_theory_R, A_plot_R), 'r-',  'LineWidth', 1., 'DisplayName','Theory cubic fit');
plot(A_plot_R, polyval(p_actual_R, A_plot_R), 'b--', 'LineWidth', 1., 'DisplayName','Actual cubic fit');

xlim([domain_min, domain_max]);
legend('Location','best');

%% === Input: Measured left & right data ===
right_data_x = [40,35,30,25,20,15,10,5,0,-5,-10,-15,-20,-25,-30,-35,-40];
right_data_y = [-44,-38,-32,-26,-21,-16,-10,-5,0,5,10,15,19,22,25,26,27];
left_data_x  = [40,35,30,25,20,15,10,5,0,-5,-10,-15,-20,-25,-30,-35,-40];
left_data_y  = [-25,-25,-22,-20,-16,-12,-8,-4,0,4,9,15,19,25,32,38,43];

% Check consistency
if ~isequal(left_data_x, right_data_x)
    error('Left and right X arrays must match.');
end

%% === Step 1: Actual average from measured points ===
avg_actual_y = (left_data_y + right_data_y) / 2;
avg_actual_x = left_data_x;

% Fit cubic
order = 3;
p_actual_avg = polyfit(avg_actual_x, avg_actual_y, order);

%% === Step 2: Theory average from theory fits ===
% p_theory and p_theory_R must be defined from your earlier cubic fits for left/right theory
p_theory_avg = (p_theory + p_theory_R) / 2;

%% === Step 3: Symbolic functions ===
syms A_sym
theory_avg_expr = poly2sym(p_theory_avg, A_sym);
actual_avg_expr = poly2sym(p_actual_avg, A_sym);

fprintf('\n=== Theoretical Average Function (Left+Right)/2 ===\n');
disp(theory_avg_expr);
fprintf('\n=== Actual Average Function (from measured data) ===\n');
disp(actual_avg_expr);

%% === Step 4: Evaluate from -25 to 25 (gap = 1) ===
A_deg = -25:1:25;
theory_Avg_deg = polyval(p_theory_avg, A_deg);
actual_Avg_deg = polyval(p_actual_avg, A_deg);

%% === Step 5: Errors ===
err = actual_Avg_deg - theory_Avg_deg;
MAE  = mean(abs(err));
RMSE = sqrt(mean(err.^2));
MaxE = max(abs(err));
MPE  = mean(abs(err) ./ max(1e-9, abs(theory_Avg_deg)) * 100);

fprintf('\n=== Error metrics on A = -25:1:25 ===\n');
fprintf('  MAE  = %.3f deg\n', MAE);
fprintf('  RMSE = %.3f deg\n', RMSE);
fprintf('  MaxE = %.3f deg\n', MaxE);
fprintf('  Mean %% Error = %.3f %%\n', MPE);

%% === Step 6: Output table (A_deg + theory_Avg_deg) ===
TheoryAvgTable = table(A_deg', theory_Avg_deg', ...
    'VariableNames', {'A_deg','theory_Avg_deg'});
disp(TheoryAvgTable);

%% === Plot: curves + ONLY actual average points ===
A_dense = linspace(-25, 25, 800);
figure; hold on; grid on;
plot(A_dense, polyval(p_theory_avg, A_dense), 'r-',  'LineWidth', 1, 'DisplayName','Theory Avg Curve');
plot(A_dense, polyval(p_actual_avg, A_dense), 'b--', 'LineWidth', 1, 'DisplayName','Actual Avg Cubic Fit');

xlabel('Steering Angle A (deg)');
ylabel('Average Motor Angle (deg)');
title('Average of Left & Right: Theory vs Actual (−25°..25°)');
legend('Location','best'); xlim([-25 25]);
