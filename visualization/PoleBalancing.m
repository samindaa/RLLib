% Pole balancing plant test with C++

clear all; close all;

sigma_0 = 0.1*ones(4, 1);
mu_0 = zeros(4, 1);
Sigma_0 = diag(sigma_0);

tau = 1.0/60;
veta = 13.2;
R = 0.01;
g = 9.81;

A = [1 tau 0 0; 0 1 0 0; 0 0 1 tau; 0 0 veta * tau 1];

b = [0; tau; 0; veta * tau / g];

Q = diag([1.25, 1, 12, 0.25]);

Sigma_T = 0.01 * Sigma_0;

x = [];

%K = [5.71; 11.3; -82.1; -21.6];
K = [10; 15; -90; -25];
X = [];
U = [];
Rt = [];
for i = 1 : 1
   % initialize
   sprintf('*** start ***')
   x = mvnrnd(mu_0, Sigma_0)'
   X = [X x];
   while  ~(abs(x(1)) > 1.5 || abs(x(3)) >= pi/6),
       u = K'*x + randn()*0.1 % random action
       r_xt_ut = x'*Q*x + u'*R*u;
       U = [U; u];
       mu = A*x + b*u;
       x = mvnrnd(mu, Sigma_T)'
       X = [X x];
       Rt = [Rt r_xt_ut];
   end   
end
figure; hold on;
subplot(3, 1, 1);plot(X(1,:));
subplot(3, 1, 2);plot(X(3,:)*180/pi);
subplot(3, 1, 3);plot(Rt);
