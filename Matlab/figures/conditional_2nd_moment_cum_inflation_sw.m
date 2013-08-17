function [c1m,c2m] = conditional_2nd_moment_cum_inflation_sw(mu,r,q,var_r,var_q,HZN)

% [c1m,c2m] = conditional_2nd_moment_cum_inflation_sw(mu,r,q,var_r,var_q,HZN)

% c1m = h-step ahead first moment for cumulative inflation, SW UC model
% c2m = h-step ahead second moment for cumulative inflation, SW UC model
% mu is the random-walk component at date t
% r,q are date-t values for the stochastic volatilities of the measurement
% and state innovations, respectively
% var_r, var_q are the variances of the respective log-volatility
% innovations
% HZN is the longest forecast horizon.
% c2m returns a HZNx1 vector of conditional variances

h = 1:1:HZN;
rr = r*exp(h'*var_r/2);
rsum = cumsum(rr);
qq = zeros(HZN,1);
for kk = 1:HZN,
    qq(kk) = q*(kk^2)*exp(kk*var_q/2);
end
qsum = cumsum(qq);
c1m = mu*h';
c2m =  (mu*h').^2 + rsum + qsum;
end

