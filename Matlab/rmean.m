function rm = rmean(x);

% this computes the recursive mean for a matrix x

[N,NG] = size(x);
rm = zeros(NG,N);
rm(1,:) = x(:,1)';
for i = 2:NG,
   rm(i,:) = rm(i-1,:) + (1/i)*(x(:,i)' - rm(i-1,:));
end

