% posterior predictive simulation
clear

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% load data, partition sample
% A = xlsread('C:\paolo2\programs\UK\UKdata','Price Data','c2:c804');
% y =(log(A(2:end))-log(A(1:end-1)));
% [T,N] = size(y); 
% date = 1210 + [0:1:T-1]';
% 
% L = 0; % VAR lag order 
% Y = y(1+L:T);
% X(1:T-L,1) = ones(T-L,1);
%   
% % partitioning the data
% Y0 = Y(514:581)';  % 1723-1790 (same as for US)
% YS = Y(582:T-L)'; % 1791-2011 (same as for US)
% date = 1791 + [0:1:size(YS,2)-1]'; % sample runs from 1791-2011

%years = 1800, 1825, 1850, 1875, 1896, 1913, 1930, 1947, 1960, 1978, 1998, 2011

NY = 12;
HZN = 10;
EC1M = zeros(HZN,NY); % posterior mean of conditional means
EC2M = zeros(HZN,NY); % posterior mean of conditional second moments

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% catalog of data files
DFILE(1,:) = ['swuc_swrp_01'];
DFILE(2,:) = ['swuc_swrp_02'];
DFILE(3,:) = ['swuc_swrp_03'];
DFILE(4,:) = ['swuc_swrp_04'];
DFILE(5,:) = ['swuc_swrp_05'];
DFILE(6,:) = ['swuc_swrp_06'];
DFILE(7,:) = ['swuc_swrp_07'];
DFILE(8,:) = ['swuc_swrp_08'];
DFILE(9,:) = ['swuc_swrp_09'];
DFILE(10,:) = ['swuc_swrp_10'];
DFILE(11,:) = ['swuc_swrp_11'];
DFILE(12,:) = ['swuc_swrp_12'];
DFILE(13,:) = ['swuc_swrp_13'];
DFILE(14,:) = ['swuc_swrp_14'];
DFILE(15,:) = ['swuc_swrp_15'];
DFILE(16,:) = ['swuc_swrp_16'];
DFILE(17,:) = ['swuc_swrp_17'];
DFILE(18,:) = ['swuc_swrp_18'];
DFILE(19,:) = ['swuc_swrp_19'];
DFILE(20,:) = ['swuc_swrp_20'];
NF = size(DFILE,1);
NB = 11;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'1800'
cd C:\paolo2\programs\UK\SWR_COL\1800
% compute SA
load(DFILE(NB,:),'SD')
[P,K,T] = size(SD);
SA2 = zeros((NF-NB+1)*P,T); % \mu
SA2(1:P,:,:) = squeeze(SD(:,2,:));
clear SD 
for i = NB+1:NF,
  j = i - NB + 1; 
  load(DFILE(i,:),'SD')
  SA2((j-1)*P+1:j*P,:,:) = squeeze(SD(:,2,:));
  clear SD;
end
%  compute RA
load(DFILE(NB,:),'RD')
[T,P] = size(RD);
RA = zeros(T,(NF-NB+1)*P);
RA(:,1:P) = RD;
clear RD 
for i = NB+1:NF,
  j = i - NB + 1; 
  load(DFILE(i,:),'RD')
  RA(:,(j-1)*P+1:j*P) = RD;
  clear RD;
end
%  compute QA
load(DFILE(NB,:),'QD')
[T,P] = size(QD);
QA = zeros(T,(NF-NB+1)*P);
QA(:,1:P) = QD;
clear QD 
for i = NB+1:NF,
  j = i - NB + 1; 
  load(DFILE(i,:),'QD')
  QA(:,(j-1)*P+1:j*P) = QD;
  clear QD;
end
% compute VA
load(DFILE(NB,:),'VD')
[P,N] = size(VD);
VA = zeros(N,(NF-NB+1)*P);
VA(:,1:P) = VD';
clear VD 
for i = NB+1:NF,
  j = i - NB + 1; 
  load(DFILE(i,:),'VD')
  VA(:,(j-1)*P+1:j*P) = VD';
  clear VD;
end
NMC = size(RA,2);
c2m = zeros(HZN,NMC);
cd C:\paolo2\programs\UK\SWR_COL\fanchart
matlabpool local 4
parfor ii = 1:NMC,
    [c1m(:,ii),c2m(:,ii)] = conditional_2nd_moment_cum_inflation_sw(SA2(ii,T-1),RA(T,ii),QA(T,ii),VA(1,ii)^2,VA(2,ii)^2,HZN);
end
matlabpool close
EC1M(:,1) = mean(c1m,2);
EC2M(:,1) = mean(c2m,2);
clear SA RA QA VA c1m c2m

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'1825'
cd C:\paolo2\programs\UK\SWR_COL\1825
% compute SA
load(DFILE(NB,:),'SD')
[P,K,T] = size(SD);
SA2 = zeros((NF-NB+1)*P,T); % \mu
SA2(1:P,:,:) = squeeze(SD(:,2,:));
clear SD 
for i = NB+1:NF,
  j = i - NB + 1; 
  load(DFILE(i,:),'SD')
  SA2((j-1)*P+1:j*P,:,:) = squeeze(SD(:,2,:));
  clear SD;
end
%  compute RA
load(DFILE(NB,:),'RD')
[T,P] = size(RD);
RA = zeros(T,(NF-NB+1)*P);
RA(:,1:P) = RD;
clear RD 
for i = NB+1:NF,
  j = i - NB + 1; 
  load(DFILE(i,:),'RD')
  RA(:,(j-1)*P+1:j*P) = RD;
  clear RD;
end
%  compute QA
load(DFILE(NB,:),'QD')
[T,P] = size(QD);
QA = zeros(T,(NF-NB+1)*P);
QA(:,1:P) = QD;
clear QD 
for i = NB+1:NF,
  j = i - NB + 1; 
  load(DFILE(i,:),'QD')
  QA(:,(j-1)*P+1:j*P) = QD;
  clear QD;
end
% compute VA
load(DFILE(NB,:),'VD')
[P,N] = size(VD);
VA = zeros(N,(NF-NB+1)*P);
VA(:,1:P) = VD';
clear VD 
for i = NB+1:NF,
  j = i - NB + 1; 
  load(DFILE(i,:),'VD')
  VA(:,(j-1)*P+1:j*P) = VD';
  clear VD;
end
NMC = size(RA,2);
cv = zeros(HZN,NMC);
cd C:\paolo2\programs\UK\SWR_COL\fanchart
matlabpool local 4
parfor ii = 1:NMC,
[c1m(:,ii),c2m(:,ii)] = conditional_2nd_moment_cum_inflation_sw(SA2(ii,T-1),RA(T,ii),QA(T,ii),VA(1,ii)^2,VA(2,ii)^2,HZN);
end
matlabpool close
EC1M(:,2) = mean(c1m,2);
EC2M(:,2) = mean(c2m,2);
clear SA RA QA VA c1m c2m

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'1850'
cd C:\paolo2\programs\UK\SWR_COL\1850
% compute SA
load(DFILE(NB,:),'SD')
[P,K,T] = size(SD);
SA2 = zeros((NF-NB+1)*P,T); % \mu
SA2(1:P,:,:) = squeeze(SD(:,2,:));
clear SD 
for i = NB+1:NF,
  j = i - NB + 1; 
  load(DFILE(i,:),'SD')
  SA2((j-1)*P+1:j*P,:,:) = squeeze(SD(:,2,:));
  clear SD;
end
%  compute RA
load(DFILE(NB,:),'RD')
[T,P] = size(RD);
RA = zeros(T,(NF-NB+1)*P);
RA(:,1:P) = RD;
clear RD 
for i = NB+1:NF,
  j = i - NB + 1; 
  load(DFILE(i,:),'RD')
  RA(:,(j-1)*P+1:j*P) = RD;
  clear RD;
end
%  compute QA
load(DFILE(NB,:),'QD')
[T,P] = size(QD);
QA = zeros(T,(NF-NB+1)*P);
QA(:,1:P) = QD;
clear QD 
for i = NB+1:NF,
  j = i - NB + 1; 
  load(DFILE(i,:),'QD')
  QA(:,(j-1)*P+1:j*P) = QD;
  clear QD;
end
% compute VA
load(DFILE(NB,:),'VD')
[P,N] = size(VD);
VA = zeros(N,(NF-NB+1)*P);
VA(:,1:P) = VD';
clear VD 
for i = NB+1:NF,
  j = i - NB + 1; 
  load(DFILE(i,:),'VD')
  VA(:,(j-1)*P+1:j*P) = VD';
  clear VD;
end
NMC = size(RA,2);
cv = zeros(HZN,NMC);
cd C:\paolo2\programs\UK\SWR_COL\fanchart
matlabpool local 4
parfor ii = 1:NMC,
    [c1m(:,ii),c2m(:,ii)] = conditional_2nd_moment_cum_inflation_sw(SA2(ii,T-1),RA(T,ii),QA(T,ii),VA(1,ii)^2,VA(2,ii)^2,HZN);
end
matlabpool close
EC1M(:,3) = mean(c1m,2);
EC2M(:,3) = mean(c2m,2);
clear SA RA QA VA c1m c2m

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'1875'
cd C:\paolo2\programs\UK\SWR_COL\1875
% compute SA
load(DFILE(NB,:),'SD')
[P,K,T] = size(SD);
SA2 = zeros((NF-NB+1)*P,T); % \mu
SA2(1:P,:,:) = squeeze(SD(:,2,:));
clear SD 
for i = NB+1:NF,
  j = i - NB + 1; 
  load(DFILE(i,:),'SD')
  SA2((j-1)*P+1:j*P,:,:) = squeeze(SD(:,2,:));
  clear SD;
end
%  compute RA
load(DFILE(NB,:),'RD')
[T,P] = size(RD);
RA = zeros(T,(NF-NB+1)*P);
RA(:,1:P) = RD;
clear RD 
for i = NB+1:NF,
  j = i - NB + 1; 
  load(DFILE(i,:),'RD')
  RA(:,(j-1)*P+1:j*P) = RD;
  clear RD;
end
%  compute QA
load(DFILE(NB,:),'QD')
[T,P] = size(QD);
QA = zeros(T,(NF-NB+1)*P);
QA(:,1:P) = QD;
clear QD 
for i = NB+1:NF,
  j = i - NB + 1; 
  load(DFILE(i,:),'QD')
  QA(:,(j-1)*P+1:j*P) = QD;
  clear QD;
end
% compute VA
load(DFILE(NB,:),'VD')
[P,N] = size(VD);
VA = zeros(N,(NF-NB+1)*P);
VA(:,1:P) = VD';
clear VD 
for i = NB+1:NF,
  j = i - NB + 1; 
  load(DFILE(i,:),'VD')
  VA(:,(j-1)*P+1:j*P) = VD';
  clear VD;
end
NMC = size(RA,2);
cv = zeros(HZN,NMC);
cd C:\paolo2\programs\UK\SWR_COL\fanchart
matlabpool local 4
parfor ii = 1:NMC,
    [c1m(:,ii),c2m(:,ii)] = conditional_2nd_moment_cum_inflation_sw(SA2(ii,T-1),RA(T,ii),QA(T,ii),VA(1,ii)^2,VA(2,ii)^2,HZN);
end
matlabpool close
EC1M(:,4) = mean(c1m,2);
EC2M(:,4) = mean(c2m,2);
clear SA RA QA VA c1m c2m

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'1896'
cd C:\paolo2\programs\UK\SWR_COL\1896
% compute SA
load(DFILE(NB,:),'SD')
[P,K,T] = size(SD);
SA2 = zeros((NF-NB+1)*P,T); % \mu
SA2(1:P,:,:) = squeeze(SD(:,2,:));
clear SD 
for i = NB+1:NF,
  j = i - NB + 1; 
  load(DFILE(i,:),'SD')
  SA2((j-1)*P+1:j*P,:,:) = squeeze(SD(:,2,:));
  clear SD;
end
%  compute RA
load(DFILE(NB,:),'RD')
[T,P] = size(RD);
RA = zeros(T,(NF-NB+1)*P);
RA(:,1:P) = RD;
clear RD 
for i = NB+1:NF,
  j = i - NB + 1; 
  load(DFILE(i,:),'RD')
  RA(:,(j-1)*P+1:j*P) = RD;
  clear RD;
end
%  compute QA
load(DFILE(NB,:),'QD')
[T,P] = size(QD);
QA = zeros(T,(NF-NB+1)*P);
QA(:,1:P) = QD;
clear QD 
for i = NB+1:NF,
  j = i - NB + 1; 
  load(DFILE(i,:),'QD')
  QA(:,(j-1)*P+1:j*P) = QD;
  clear QD;
end
% compute VA
load(DFILE(NB,:),'VD')
[P,N] = size(VD);
VA = zeros(N,(NF-NB+1)*P);
VA(:,1:P) = VD';
clear VD 
for i = NB+1:NF,
  j = i - NB + 1; 
  load(DFILE(i,:),'VD')
  VA(:,(j-1)*P+1:j*P) = VD';
  clear VD;
end
NMC = size(RA,2);
cv = zeros(HZN,NMC);
cd C:\paolo2\programs\UK\SWR_COL\fanchart
matlabpool local 4
parfor ii = 1:NMC,
    [c1m(:,ii),c2m(:,ii)] = conditional_2nd_moment_cum_inflation_sw(SA2(ii,T-1),RA(T,ii),QA(T,ii),VA(1,ii)^2,VA(2,ii)^2,HZN);
end
matlabpool close
EC1M(:,5) = mean(c1m,2);
EC2M(:,5) = mean(c2m,2);
clear SA RA QA VA c1m c2m

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'1913'
cd C:\paolo2\programs\UK\SWR_COL\1913
% compute SA
load(DFILE(NB,:),'SD')
[P,K,T] = size(SD);
SA2 = zeros((NF-NB+1)*P,T); % \mu
SA2(1:P,:,:) = squeeze(SD(:,2,:));
clear SD 
for i = NB+1:NF,
  j = i - NB + 1; 
  load(DFILE(i,:),'SD')
  SA2((j-1)*P+1:j*P,:,:) = squeeze(SD(:,2,:));
  clear SD;
end
%  compute RA
load(DFILE(NB,:),'RD')
[T,P] = size(RD);
RA = zeros(T,(NF-NB+1)*P);
RA(:,1:P) = RD;
clear RD 
for i = NB+1:NF,
  j = i - NB + 1; 
  load(DFILE(i,:),'RD')
  RA(:,(j-1)*P+1:j*P) = RD;
  clear RD;
end
%  compute QA
load(DFILE(NB,:),'QD')
[T,P] = size(QD);
QA = zeros(T,(NF-NB+1)*P);
QA(:,1:P) = QD;
clear QD 
for i = NB+1:NF,
  j = i - NB + 1; 
  load(DFILE(i,:),'QD')
  QA(:,(j-1)*P+1:j*P) = QD;
  clear QD;
end
% compute VA
load(DFILE(NB,:),'VD')
[P,N] = size(VD);
VA = zeros(N,(NF-NB+1)*P);
VA(:,1:P) = VD';
clear VD 
for i = NB+1:NF,
  j = i - NB + 1; 
  load(DFILE(i,:),'VD')
  VA(:,(j-1)*P+1:j*P) = VD';
  clear VD;
end
NMC = size(RA,2);
cv = zeros(HZN,NMC);
cd C:\paolo2\programs\UK\SWR_COL\fanchart
matlabpool local 4
parfor ii = 1:NMC,
[c1m(:,ii),c2m(:,ii)] = conditional_2nd_moment_cum_inflation_sw(SA2(ii,T-1),RA(T,ii),QA(T,ii),VA(1,ii)^2,VA(2,ii)^2,HZN);
end
matlabpool close
EC1M(:,6) = mean(c1m,2);
EC2M(:,6) = mean(c2m,2);
clear SA RA QA VA c1m c2m

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'1930'
cd C:\paolo2\programs\UK\SWR_COL\1930
% compute SA
load(DFILE(NB,:),'SD')
[P,K,T] = size(SD);
SA2 = zeros((NF-NB+1)*P,T); % \mu
SA2(1:P,:,:) = squeeze(SD(:,2,:));
clear SD 
for i = NB+1:NF,
  j = i - NB + 1; 
  load(DFILE(i,:),'SD')
  SA2((j-1)*P+1:j*P,:,:) = squeeze(SD(:,2,:));
  clear SD;
end
%  compute RA
load(DFILE(NB,:),'RD')
[T,P] = size(RD);
RA = zeros(T,(NF-NB+1)*P);
RA(:,1:P) = RD;
clear RD 
for i = NB+1:NF,
  j = i - NB + 1; 
  load(DFILE(i,:),'RD')
  RA(:,(j-1)*P+1:j*P) = RD;
  clear RD;
end
%  compute QA
load(DFILE(NB,:),'QD')
[T,P] = size(QD);
QA = zeros(T,(NF-NB+1)*P);
QA(:,1:P) = QD;
clear QD 
for i = NB+1:NF,
  j = i - NB + 1; 
  load(DFILE(i,:),'QD')
  QA(:,(j-1)*P+1:j*P) = QD;
  clear QD;
end
% compute VA
load(DFILE(NB,:),'VD')
[P,N] = size(VD);
VA = zeros(N,(NF-NB+1)*P);
VA(:,1:P) = VD';
clear VD 
for i = NB+1:NF,
  j = i - NB + 1; 
  load(DFILE(i,:),'VD')
  VA(:,(j-1)*P+1:j*P) = VD';
  clear VD;
end
NMC = size(RA,2);
cv = zeros(HZN,NMC);
cd C:\paolo2\programs\UK\SWR_COL\fanchart
matlabpool local 4
parfor ii = 1:NMC,
[c1m(:,ii),c2m(:,ii)] = conditional_2nd_moment_cum_inflation_sw(SA2(ii,T-1),RA(T,ii),QA(T,ii),VA(1,ii)^2,VA(2,ii)^2,HZN);
end
matlabpool close
EC1M(:,7) = mean(c1m,2);
EC2M(:,7) = mean(c2m,2);
clear SA RA QA VA c1m c2m

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'1947'
cd C:\paolo2\programs\UK\SWR_COL\1947
% compute SA
load(DFILE(NB,:),'SD')
[P,K,T] = size(SD);
SA2 = zeros((NF-NB+1)*P,T); % \mu
SA2(1:P,:,:) = squeeze(SD(:,2,:));
clear SD 
for i = NB+1:NF,
  j = i - NB + 1; 
  load(DFILE(i,:),'SD')
  SA2((j-1)*P+1:j*P,:,:) = squeeze(SD(:,2,:));
  clear SD;
end
%  compute RA
load(DFILE(NB,:),'RD')
[T,P] = size(RD);
RA = zeros(T,(NF-NB+1)*P);
RA(:,1:P) = RD;
clear RD 
for i = NB+1:NF,
  j = i - NB + 1; 
  load(DFILE(i,:),'RD')
  RA(:,(j-1)*P+1:j*P) = RD;
  clear RD;
end
%  compute QA
load(DFILE(NB,:),'QD')
[T,P] = size(QD);
QA = zeros(T,(NF-NB+1)*P);
QA(:,1:P) = QD;
clear QD 
for i = NB+1:NF,
  j = i - NB + 1; 
  load(DFILE(i,:),'QD')
  QA(:,(j-1)*P+1:j*P) = QD;
  clear QD;
end
% compute VA
load(DFILE(NB,:),'VD')
[P,N] = size(VD);
VA = zeros(N,(NF-NB+1)*P);
VA(:,1:P) = VD';
clear VD 
for i = NB+1:NF,
  j = i - NB + 1; 
  load(DFILE(i,:),'VD')
  VA(:,(j-1)*P+1:j*P) = VD';
  clear VD;
end
NMC = size(RA,2);
cv = zeros(HZN,NMC);
cd C:\paolo2\programs\UK\SWR_COL\fanchart
matlabpool local 4
parfor ii = 1:NMC,
    [c1m(:,ii),c2m(:,ii)] = conditional_2nd_moment_cum_inflation_sw(SA2(ii,T-1),RA(T,ii),QA(T,ii),VA(1,ii)^2,VA(2,ii)^2,HZN);
end
matlabpool close
EC1M(:,8) = mean(c1m,2);
EC2M(:,8) = mean(c2m,2);
clear SA RA QA VA c1m c2m

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'1960'
cd C:\paolo2\programs\UK\SWR_COL\1960
% compute SA
load(DFILE(NB,:),'SD')
[P,K,T] = size(SD);
SA2 = zeros((NF-NB+1)*P,T); % \mu
SA2(1:P,:,:) = squeeze(SD(:,2,:));
clear SD 
for i = NB+1:NF,
  j = i - NB + 1; 
  load(DFILE(i,:),'SD')
  SA2((j-1)*P+1:j*P,:,:) = squeeze(SD(:,2,:));
  clear SD;
end
%  compute RA
load(DFILE(NB,:),'RD')
[T,P] = size(RD);
RA = zeros(T,(NF-NB+1)*P);
RA(:,1:P) = RD;
clear RD 
for i = NB+1:NF,
  j = i - NB + 1; 
  load(DFILE(i,:),'RD')
  RA(:,(j-1)*P+1:j*P) = RD;
  clear RD;
end
%  compute QA
load(DFILE(NB,:),'QD')
[T,P] = size(QD);
QA = zeros(T,(NF-NB+1)*P);
QA(:,1:P) = QD;
clear QD 
for i = NB+1:NF,
  j = i - NB + 1; 
  load(DFILE(i,:),'QD')
  QA(:,(j-1)*P+1:j*P) = QD;
  clear QD;
end
% compute VA
load(DFILE(NB,:),'VD')
[P,N] = size(VD);
VA = zeros(N,(NF-NB+1)*P);
VA(:,1:P) = VD';
clear VD 
for i = NB+1:NF,
  j = i - NB + 1; 
  load(DFILE(i,:),'VD')
  VA(:,(j-1)*P+1:j*P) = VD';
  clear VD;
end
NMC = size(RA,2);
cv = zeros(HZN,NMC);
cd C:\paolo2\programs\UK\SWR_COL\fanchart
matlabpool local 4
parfor ii = 1:NMC,
    [c1m(:,ii),c2m(:,ii)] = conditional_2nd_moment_cum_inflation_sw(SA2(ii,T-1),RA(T,ii),QA(T,ii),VA(1,ii)^2,VA(2,ii)^2,HZN);
end
matlabpool close
EC1M(:,9) = mean(c1m,2);
EC2M(:,9) = mean(c2m,2);
clear SA RA QA VA c1m c2m

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'1978'
cd C:\paolo2\programs\UK\SWR_COL\1978
% compute SA
load(DFILE(NB,:),'SD')
[P,K,T] = size(SD);
SA2 = zeros((NF-NB+1)*P,T); % \mu
SA2(1:P,:,:) = squeeze(SD(:,2,:));
clear SD 
for i = NB+1:NF,
  j = i - NB + 1; 
  load(DFILE(i,:),'SD')
  SA2((j-1)*P+1:j*P,:,:) = squeeze(SD(:,2,:));
  clear SD;
end
%  compute RA
load(DFILE(NB,:),'RD')
[T,P] = size(RD);
RA = zeros(T,(NF-NB+1)*P);
RA(:,1:P) = RD;
clear RD 
for i = NB+1:NF,
  j = i - NB + 1; 
  load(DFILE(i,:),'RD')
  RA(:,(j-1)*P+1:j*P) = RD;
  clear RD;
end
%  compute QA
load(DFILE(NB,:),'QD')
[T,P] = size(QD);
QA = zeros(T,(NF-NB+1)*P);
QA(:,1:P) = QD;
clear QD 
for i = NB+1:NF,
  j = i - NB + 1; 
  load(DFILE(i,:),'QD')
  QA(:,(j-1)*P+1:j*P) = QD;
  clear QD;
end
% compute VA
load(DFILE(NB,:),'VD')
[P,N] = size(VD);
VA = zeros(N,(NF-NB+1)*P);
VA(:,1:P) = VD';
clear VD 
for i = NB+1:NF,
  j = i - NB + 1; 
  load(DFILE(i,:),'VD')
  VA(:,(j-1)*P+1:j*P) = VD';
  clear VD;
end
NMC = size(RA,2);
cv = zeros(HZN,NMC);
cd C:\paolo2\programs\UK\SWR_COL\fanchart
matlabpool local 4
parfor ii = 1:NMC,
    [c1m(:,ii),c2m(:,ii)] = conditional_2nd_moment_cum_inflation_sw(SA2(ii,T-1),RA(T,ii),QA(T,ii),VA(1,ii)^2,VA(2,ii)^2,HZN);
end
matlabpool close
EC1M(:,10) = mean(c1m,2);
EC2M(:,10) = mean(c2m,2);
clear SA RA QA VA c1m c2m

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'1998'
cd C:\paolo2\programs\UK\SWR_COL\1998
% compute SA
load(DFILE(NB,:),'SD')
[P,K,T] = size(SD);
SA2 = zeros((NF-NB+1)*P,T); % \mu
SA2(1:P,:,:) = squeeze(SD(:,2,:));
clear SD 
for i = NB+1:NF,
  j = i - NB + 1; 
  load(DFILE(i,:),'SD')
  SA2((j-1)*P+1:j*P,:,:) = squeeze(SD(:,2,:));
  clear SD;
end
%  compute RA
load(DFILE(NB,:),'RD')
[T,P] = size(RD);
RA = zeros(T,(NF-NB+1)*P);
RA(:,1:P) = RD;
clear RD 
for i = NB+1:NF,
  j = i - NB + 1; 
  load(DFILE(i,:),'RD')
  RA(:,(j-1)*P+1:j*P) = RD;
  clear RD;
end
%  compute QA
load(DFILE(NB,:),'QD')
[T,P] = size(QD);
QA = zeros(T,(NF-NB+1)*P);
QA(:,1:P) = QD;
clear QD 
for i = NB+1:NF,
  j = i - NB + 1; 
  load(DFILE(i,:),'QD')
  QA(:,(j-1)*P+1:j*P) = QD;
  clear QD;
end
% compute VA
load(DFILE(NB,:),'VD')
[P,N] = size(VD);
VA = zeros(N,(NF-NB+1)*P);
VA(:,1:P) = VD';
clear VD 
for i = NB+1:NF,
  j = i - NB + 1; 
  load(DFILE(i,:),'VD')
  VA(:,(j-1)*P+1:j*P) = VD';
  clear VD;
end
NMC = size(RA,2);
cv = zeros(HZN,NMC);
cd C:\paolo2\programs\UK\SWR_COL\fanchart
matlabpool local 4
parfor ii = 1:NMC,
    [c1m(:,ii),c2m(:,ii)] = conditional_2nd_moment_cum_inflation_sw(SA2(ii,T-1),RA(T,ii),QA(T,ii),VA(1,ii)^2,VA(2,ii)^2,HZN);
end
matlabpool close
EC1M(:,11) = mean(c1m,2);
EC2M(:,11) = mean(c2m,2);
clear SA RA QA VA c1m c2m

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'2011'
cd C:\paolo2\programs\UK\SWR_COL\2011
% compute SA
load(DFILE(NB,:),'SD')
[P,K,T] = size(SD);
SA2 = zeros((NF-NB+1)*P,T); % \mu
SA2(1:P,:,:) = squeeze(SD(:,2,:));
clear SD 
for i = NB+1:NF,
  j = i - NB + 1; 
  load(DFILE(i,:),'SD')
  SA2((j-1)*P+1:j*P,:,:) = squeeze(SD(:,2,:));
  clear SD;
end
%  compute RA
load(DFILE(NB,:),'RD')
[T,P] = size(RD);
RA = zeros(T,(NF-NB+1)*P);
RA(:,1:P) = RD;
clear RD 
for i = NB+1:NF,
  j = i - NB + 1; 
  load(DFILE(i,:),'RD')
  RA(:,(j-1)*P+1:j*P) = RD;
  clear RD;
end
%  compute QA
load(DFILE(NB,:),'QD')
[T,P] = size(QD);
QA = zeros(T,(NF-NB+1)*P);
QA(:,1:P) = QD;
clear QD 
for i = NB+1:NF,
  j = i - NB + 1; 
  load(DFILE(i,:),'QD')
  QA(:,(j-1)*P+1:j*P) = QD;
  clear QD;
end
% compute VA
load(DFILE(NB,:),'VD')
[P,N] = size(VD);
VA = zeros(N,(NF-NB+1)*P);
VA(:,1:P) = VD';
clear VD 
for i = NB+1:NF,
  j = i - NB + 1; 
  load(DFILE(i,:),'VD')
  VA(:,(j-1)*P+1:j*P) = VD';
  clear VD;
end
NMC = size(RA,2);
cv = zeros(HZN,NMC);
cd C:\paolo2\programs\UK\SWR_COL\fanchart
matlabpool local 4
parfor ii = 1:NMC,
    [c1m(:,ii),c2m(:,ii)] = conditional_2nd_moment_cum_inflation_sw(SA2(ii,T-1),RA(T,ii),QA(T,ii),VA(1,ii)^2,VA(2,ii)^2,HZN);
end
matlabpool close
EC1M(:,12) = mean(c1m,2);
EC2M(:,12) = mean(c2m,2);
clear SA RA QA VA c1m c2m

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ECV = EC2M - EC1M.^2; % conditional variance = conditional second moment minus square of conditional mean
CSTD = ECV.^.5; % conditional standard deviation
CRMS = EC2M.^.5; % conditional root-mean-square statistics


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 4x3 plot of conditional standard deviations and root-mean-square statistics
figure 
subplot(4,3,1)
plot(CRMS(:,1),'-r','Linewidth',2); hold on;
plot(CSTD(:,1),'-b','Linewidth',2); hold off;
title('1800','Fontsize',14)
legend('RMS','STD')
set(gca,'Fontsize',14)
axis([0 HZN 0 1.0])
grid

subplot(4,3,2)
plot(CRMS(:,2),'-r','Linewidth',2); hold on;
plot(CSTD(:,2),'-b','Linewidth',2); hold off;
title('1825','Fontsize',14)
set(gca,'Fontsize',14)
axis([0 HZN 0 1.0])
grid

subplot(4,3,3)
plot(CRMS(:,3),'-r','Linewidth',2); hold on;
plot(CSTD(:,3),'-b','Linewidth',2); hold off;
title('1850','Fontsize',14)
set(gca,'Fontsize',14)
axis([0 HZN 0 1.0])
grid

subplot(4,3,4)
plot(CRMS(:,4),'-r','Linewidth',2); hold on;
plot(CSTD(:,4),'-b','Linewidth',2); hold off;
title('1875','Fontsize',14)
set(gca,'Fontsize',14)
axis([0 HZN 0 1.0])
grid

subplot(4,3,5)
plot(CRMS(:,5),'-r','Linewidth',2); hold on;
plot(CSTD(:,5),'-b','Linewidth',2); hold off;
title('1896','Fontsize',14)
set(gca,'Fontsize',14)
axis([0 HZN 0 1.0])
grid

subplot(4,3,6)
plot(CRMS(:,6),'-r','Linewidth',2); hold on;
plot(CSTD(:,6),'-b','Linewidth',2); hold off;
title('1913','Fontsize',14)
set(gca,'Fontsize',14)
axis([0 HZN 0 1.0])
grid

subplot(4,3,7)
plot(CRMS(:,7),'-r','Linewidth',2); hold on;
plot(CSTD(:,7),'-b','Linewidth',2); hold off;
title('1930','Fontsize',14)
set(gca,'Fontsize',14)
axis([0 HZN 0 1.0])
grid

subplot(4,3,8)
plot(CRMS(:,8),'-r','Linewidth',2); hold on;
plot(CSTD(:,8),'-b','Linewidth',2); hold off;
title('1947','Fontsize',14)
set(gca,'Fontsize',14)
axis([0 HZN 0 1.0])
grid

subplot(4,3,9)
plot(CRMS(:,9),'-r','Linewidth',2); hold on;
plot(CSTD(:,9),'-b','Linewidth',2); hold off;
title('1960','Fontsize',14)
set(gca,'Fontsize',14)
axis([0 HZN 0 1.0])
grid

subplot(4,3,10)
plot(CRMS(:,10),'-r','Linewidth',2); hold on;
plot(CSTD(:,10),'-b','Linewidth',2); hold off;
title('1978','Fontsize',14)
set(gca,'Fontsize',14)
axis([0 HZN 0 1.0])
grid
xlabel('Years ahead','Fontsize',14)

subplot(4,3,11)
plot(CRMS(:,11),'-r','Linewidth',2); hold on;
plot(CSTD(:,11),'-b','Linewidth',2); hold off;
title('1998','Fontsize',14)
set(gca,'Fontsize',14)
axis([0 HZN 0 1.0])
grid
xlabel('Years ahead','Fontsize',14)

subplot(4,3,12)
plot(CRMS(:,12),'-r','Linewidth',2); hold on;
plot(CSTD(:,12),'-b','Linewidth',2); hold off;
title('2011','Fontsize',14)
set(gca,'Fontsize',14)
axis([0 HZN 0 1.0])
grid
xlabel('Years ahead','Fontsize',14)


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% most predictable: 1913 v. 1960, 1978, 1998, 2011
figure
subplot(2,2,1)
plot(CSTD(:,6),'-b','Linewidth',2); hold on;
plot(CSTD(:,9),'-r','Linewidth',2); hold off;
legend('1913','1960','Location','NorthWest')
ylabel('Standard Deviation','Fontsize',14)
set(gca,'Fontsize',14)
axis([0 HZN 0 0.25])
grid

subplot(2,2,2)
plot(CSTD(:,6),'-b','Linewidth',2); hold on;
plot(CSTD(:,10),'-r','Linewidth',2); hold off;
legend('1913','1978','Location','NorthWest')
set(gca,'Fontsize',14)
axis([0 HZN 0 0.25])
grid

subplot(2,2,3)
plot(CSTD(:,6),'-b','Linewidth',2); hold on;
plot(CSTD(:,11),'-r','Linewidth',2); hold off;
legend('1913','1998','Location','NorthWest')
ylabel('Standard Deviation','Fontsize',14)
set(gca,'Fontsize',14)
axis([0 HZN 0 0.25])
xlabel('Years Ahead','Fontsize',14)
grid

subplot(2,2,4)
plot(CSTD(:,6),'-b','Linewidth',2); hold on;
plot(CSTD(:,12),'-r','Linewidth',2); hold off;
legend('1913','2011','Location','NorthWest')
set(gca,'Fontsize',14)
axis([0 HZN 0 0.25])
xlabel('Years Ahead','Fontsize',14)
grid

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 2011 v preWWI
figure
subplot(2,2,1)
plot(CSTD(:,12),'-b','Linewidth',2); hold on;
plot(CSTD(:,3),'-r','Linewidth',2); hold off;
legend('2011','1850','Location','NorthWest')
ylabel('Standard Deviation','Fontsize',14)
set(gca,'Fontsize',14)
axis([0 HZN 0 0.5])
grid

subplot(2,2,2)
plot(CSTD(:,12),'-b','Linewidth',2); hold on;
plot(CSTD(:,4),'-r','Linewidth',2); hold off;
legend('2011','1875','Location','NorthWest')
set(gca,'Fontsize',14)
axis([0 HZN 0 0.5])
grid

subplot(2,2,3)
plot(CSTD(:,12),'-b','Linewidth',2); hold on;
plot(CSTD(:,5),'-r','Linewidth',2); hold off;
legend('2011','1896','Location','NorthWest')
ylabel('Standard Deviation','Fontsize',14)
set(gca,'Fontsize',14)
axis([0 HZN 0 0.5])
xlabel('Years Ahead','Fontsize',14)
grid

subplot(2,2,4)
plot(CSTD(:,12),'-b','Linewidth',2); hold on;
plot(CSTD(:,6),'-r','Linewidth',2); hold off;
legend('2011','1913','Location','NorthWest')
set(gca,'Fontsize',14)
axis([0 HZN 0 0.5])
xlabel('Years Ahead','Fontsize',14)
grid

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% RMS: 1800-1978, 1913, 1960, 2011
figure 
subplot(2,2,1)
plot(CRMS(:,1),'-b','Linewidth',2); hold on;
plot(CRMS(:,10),'-r','Linewidth',2); hold off;
legend('1800','1978','Location','NorthWest')
set(gca,'Fontsize',14)
axis([0 HZN 0 1.2])
xlabel('Years Ahead','Fontsize',14)
ylabel('Root-mean-square','Fontsize',14)
grid

subplot(2,2,2)
plot(CRMS(:,6),'-b','Linewidth',2); hold on;
plot(CRMS(:,9),'-g','Linewidth',2); hold on;
plot(CRMS(:,12),'-r','Linewidth',2); hold off;
legend('1913','1960','2011','Location','NorthWest')
set(gca,'Fontsize',14)
axis([0 HZN 0 0.4])
xlabel('Years Ahead','Fontsize',14)
grid


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% RMS: 1960, 2011 v. 1850, 1875, 1896, 1913
figure
subplot(3,2,1)
plot(CRMS(:,9),'-b','Linewidth',2); hold on;
plot(CRMS(:,3),'-r','Linewidth',2); hold off;
legend('1960','1850','Location','NorthWest')
ylabel('RMS','Fontsize',14)
set(gca,'Fontsize',14)
axis([0 HZN 0 0.5])
grid

subplot(3,2,2)
plot(CRMS(:,12),'-b','Linewidth',2); hold on;
plot(CRMS(:,3),'-r','Linewidth',2); hold off;
legend('2011','1850','Location','NorthWest')
set(gca,'Fontsize',14)
axis([0 HZN 0 0.5])
grid

subplot(3,2,3)
plot(CRMS(:,9),'-b','Linewidth',2); hold on;
plot(CRMS(:,4),'-r','Linewidth',2); hold off;
legend('1960','1875','Location','NorthWest')
ylabel('RMS','Fontsize',14)
set(gca,'Fontsize',14)
axis([0 HZN 0 0.5])
grid

subplot(3,2,4)
plot(CRMS(:,12),'-b','Linewidth',2); hold on;
plot(CRMS(:,4),'-r','Linewidth',2); hold off;
legend('2011','1875','Location','NorthWest')
set(gca,'Fontsize',14)
axis([0 HZN 0 0.5])
grid

subplot(3,2,5)
plot(CRMS(:,9),'-b','Linewidth',2); hold on;
plot(CRMS(:,5),'-r','Linewidth',2); hold off;
legend('1960','1896','Location','NorthWest')
ylabel('RMS','Fontsize',14)
set(gca,'Fontsize',14)
axis([0 HZN 0 0.5])
xlabel('Years ahead')
grid

subplot(3,2,6)
plot(CRMS(:,12),'-b','Linewidth',2); hold on;
plot(CRMS(:,5),'-r','Linewidth',2); hold off;
legend('2011','1896','Location','NorthWest')
set(gca,'Fontsize',14)
axis([0 HZN 0 0.5])
xlabel('Years ahead')
grid





