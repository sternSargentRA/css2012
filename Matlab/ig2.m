function [v,v1,d1] = ig2(v0,d0,x);

% [v,v1,d1] = ig2s(v0,d0,x);

% This file returns posterior draw, v, from an inverse gamma with prior degrees of
% freedom v0/2 and scale parameter d0/2.  The posterior values are v1 and d1, respectively.
% x is a vector of innovations.

% The simulation method follows bauwens, et al p 317.  IG2(s,v)
%      simulate x = chisquare(v)
%      deliver s/x
global rand_ind randoms;

T = size(x,1);
v1 = v0 + T;
d1 = d0 + x'*x;
% z = randn(v1,1);
z = reshape(randoms(rand_ind:rand_ind+v1 - 1), v1, 1);
rand_ind = rand_ind + v1;
x = z'*z;
v = d1/x;
end

