function [S0,P0,P1] = kf_SWR(Y,Q,R,Sm,SI,PI,T);

% [S0,P0,P1] = kf_SWR(Y,Q,R,Sm,SI,PI,T);

% This file performs the forward kalman filter recursions for the Stock-Watson-Romer model.  
% Y is inflation
% Q,R are the SW state innovation variances
% Sm is the standard deviation of the measurement error
% SI,PI are the initial values for the recursions, S(1|0) and P(1|0)
% T is the sample size

S0 = zeros(2,T); % current estimate of the state, S(t|t)
S1 = zeros(2,T); % one-step ahead estimate of the state, S(t+1|t)
P0 = zeros(2,2,T); % current estimate of the covariance matrix, P(t|t)
P1 = zeros(2,2,T); % one-step ahead covariance matrix, P(t+1|t)

% constant parameters
A = [0 1; 0 1];
C = [1 0];

% date 1
y10 = C*SI; % E(y(t|t-1)
D = Sm(1);
V10 = C*PI*C' + D*D'; % V(y(t|t-1)
S0(:,1) = SI + PI*C'*inv(V10)*( Y(:,1) - y10 ); % E(S(t|t))
P0(:,:,1) = PI - PI*C'*inv(V10)*C*PI; % V(S(t|t))
S1(:,1) = A*S0(:,1); % E(S(t+1|t)
B = [R(2)^.5 Q(2)^.5; 0 Q(2)^.5];
P1(:,:,1) = A*P0(:,:,1)*A' + B*B'; % V(S(t+1|t)

% Iterating through the rest of the sample
for i = 2:T,
  y10 = C*S1(:,i-1); % E(y(t|t-1)
  D = Sm(i);
  V10 = C*P1(:,:,i-1)*C' + D*D'; % V(y(t|t-1)
  S0(:,i) = S1(:,i-1) + P1(:,:,i-1)*C'*inv(V10)*( Y(:,i) - y10 ); % E(S(t|t))
  P0(:,:,i) = P1(:,:,i-1) - P1(:,:,i-1)*C'*inv(V10)*C*P1(:,:,i-1); % V(S(t|t))
  S1(:,i) = A*S0(:,i); % E(S(t+1|t)
  B = [R(i+1)^.5 Q(i+1)^.5; 0 Q(i+1)^.5];
  P1(:,:,i) = A*P0(:,:,i)*A' + B*B'; % V(S(t+1|t)
end