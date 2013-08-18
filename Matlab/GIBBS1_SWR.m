function SA = GIBBS1_SWR(S0,P0,P1,T);

global rand_ind randoms;

% function SA = GIBBS1_SWR(S0,P0,P1,T);

% This file executes the Carter-Kohn backward sampler for the
% Stock-Watson-Romer model.

% S0,P0,P1 are outputs of the forward Kalman filter
A = [0 1; 0 1];

% initialize arrays for Gibbs sampler
SA = zeros(2,T); % artificial states
SM = zeros(2,1); % backward update for conditional mean of state vector
PM = zeros(2,2); % backward update for projection matrix
P = zeros(2,2); % backward update for conditional variance matrix

wa = reshape(randoms(rand_ind:rand_ind + (2 * T) - 1), 2, T);
rand_ind = rand_ind + (2 * T);

% wa = randn(2,T); % draws for state innovations

% Backward recursions and sampling
% Terminal state
SA(:,T) = S0(:,T) + real(sqrtm(P0(:,:,T)))*wa(:,T);

% iterating back through the rest of the sample
for i = 1:T-1,
   PM = P0(:,:,T-i)*A'*inv(P1(:,:,T-i));
   P = P0(:,:,T-i) - PM*A*P0(:,:,T-i);
   SM = S0(:,T-i) + PM*(SA(:,T-i+1) - A*S0(:,T-i));
   SA(:,T-i) = SM + real(sqrtm(P))*wa(:,T-i);
end

1;
