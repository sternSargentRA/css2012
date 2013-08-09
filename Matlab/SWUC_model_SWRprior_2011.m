%  This simulates the posterior for the Stock-Watson-Romer UC model with
%  stochastic volatility and measurement error.  For the UK, two break
%  dates for measurement error are assumed, 1791-1914 and 1915-1947.

clear
NG = 100000; % number of draws from Gibbs sampler per data file
NF = 20;

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

% variables to be saved
varname(1,:) = ['SD']; % states
varname(2,:) = ['QD']; % variance of permanent innovation
varname(3,:) = ['RD']; % svol for transient innovation
varname(4,:) = ['VD']; % standard deviation of volatility innovations
varname(5,:) = ['MD']; % standard deviation for measurement error

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% load data, partition sample
% A = xlsread('../data/UKdata','Price Data','c2:c804'); % global financial database
load ../data/UKdata.txt
A = UKdata(:, 2);
y =(log(A(2:end))-log(A(1:end-1)));
[T,N] = size(y);
date = 1210 + [0:1:T-1]';

% splicing the data
Y0 = y(512:581);  % 1721-1790 training sample
YS_1948_2011 = y(739:802); % 1948-2011
clear A y

load ../data/Lindert_Williamson.txt -ascii % Lindert-Williamson (B.R. Mitchell, British Historical Statistics)
lnP = log(Lindert_Williamson(:,2));
YS_1791_1850 = diff(lnP);
clear Lindert_Williamson lnP

load ../data/Bowley.txt -ascii % Bowley (B.R. Mitchell, British Historical Statistics)
lnP = log(Bowley(:,2));
YS_1847_1914 = diff(lnP);
clear Bowley lnP

load ../data/LaborDepartment.txt -ascii % Labor department (B.R. Mitchell, British Historical Statistics)
lnP = log(LaborDepartment(:,2));
YS_1915_1947 = diff(lnP);
clear LaborDepartment lnP

% spliced data: L-W, B, LD, GFD
y = [YS_1791_1850; YS_1847_1914(5:end); YS_1915_1947; YS_1948_2011];
T = size(y,1);
date = 1791 + [0:1:T-1]';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
L = 0; % VAR lag order
YS = y(1+L:T)';
X(1:T-L,1) = ones(T-L,1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% a weakly informative prior
SI = mean(Y0)*ones(2,1); % prior mean on initial value of state; first element is \pi
PI = [.15^2 0; 0 0.025^2]; % prior variance on initial state

R0 = var(Y0); % prior variance for SW transient innovations
Q0 = R0/25; % prior variance for trend innovations

df = 2; % prior degrees of freedom

% clear initial sample
clear y Y X Y0 X0
[T,N] = size(YS');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% priors for sv (inverse gamma) (standard dev for volatility innovation)
v0 = 10;
svr0 = 0.2236*sqrt((v0+1)/v0); % stock and watson's calibrated value adjusted for time aggregation
svq0 = 0.2236*sqrt((v0+1)/v0);
dr0 = v0*(svr0^2);
dq0 = v0*(svq0^2);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% prior variance for log R0, log Q0 (ballpark numbers)
ss0 = 5;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% prior for measurement-error variance \sigma_m (prior is same for both
% periods)
vm0 = 7;
sm0 = 0.5*sqrt(R0)*sqrt((vm0+1)/vm0);
dm0 = vm0*(sm0^2);
sm_post_48 = 0.0001; % after 1948, the measurement error has a standard deviation of 1 basis point. This is just to simplify programming

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% initialize gibbs arrays
SA = zeros(NG,2,T); % draws of the state vector
QA = zeros(T+1,NG); % stochastic volatilities for SW permanent innovation
RA = zeros(T+1,NG); % stochastic volatilities for SW transient innovation
SV = zeros(NG,2); % standard error for log volatility innovations
SMV = zeros(NG,3); % standard error for measurement error

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% initialize stochastic volatilities and measurement error variance
QA(:,1) = Q0*ones(T+1,1);
RA(:,1) = R0*ones(T+1,1);
SV(1,:) = [svr0 svq0];
SMT(:,1) = [sm0*ones(157,1); sm_post_48*ones(size(date(158:221,1)))];
SMV(1,:) = sm0;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% begin MCMC
for file = 1:NF,
    file
    for iter = 2:NG,

       % states conditional on hyperparameters and data
       [S0,P0,P1] = kf_SWR(YS,QA(:,iter-1),RA(:,iter-1),SMT,SI,PI,T);
       SA(iter,:,:) = GIBBS1_SWR(S0,P0,P1,T);

       % stochastic volatilities
       f = diff([SI squeeze(SA(iter,:,:))]'); % SW state innovations
       % log R|sv,y
       RA(1,iter) = svmh0(RA(2,iter-1),0,1,SV(iter-1,1),log(R0),ss0);
       for t = 2:T,
          RA(t,iter) = svmh(RA(t+1,iter-1),RA(t-1,iter),0,1,SV(iter-1,1),f(t-1,1),RA(t,iter-1));
       end
       RA(T+1,iter) = svmhT(RA(T,iter),0,1,SV(iter-1,1),f(T,1),RA(T+1,iter-1));
       % log Q|sv,y
       QA(1,iter) = svmh0(QA(2,iter-1),0,1,SV(iter-1,2),log(Q0),ss0);
       for t = 2:T,
          QA(t,iter) = svmh(QA(t+1,iter-1),QA(t-1,iter),0,1,SV(iter-1,2),f(t-1,2),QA(t,iter-1));
       end
       QA(T+1,iter) = svmhT(QA(T,iter),0,1,SV(iter-1,2),f(T,2),QA(T+1,iter-1));
       % svr
       lr = log(RA(:,iter));
       er(:,1) = lr(2:T+1,1) - lr(1:T,1);  % random walk
       v = ig2(v0,dr0,er(:,1));
       SV(iter,1) = v^.5;
       % svq
       lq = log(QA(:,iter));
       eq(:,1) = lq(2:T+1,1) - lq(1:T,1);  % random walk
       v = ig2(v0,dq0,eq(:,1));
       SV(iter,2) = v^.5;

       % measurement error
       em = YS - squeeze(SA(iter,1,:))';
       v1 = ig2(vm0,dm0,em(1,1:60)'); % measurement error 1791-1850 (Lindert-Williamson)
       v2 = ig2(vm0,dm0,em(1,61:124)'); % measurement error 1851-1914 (Bowley)
       v3 = ig2(vm0,dm0,em(1,125:157)'); % measurement error 1915-1947 (Labor Department)
       SMV(iter,:) = [v1 v2 v3].^.5;
       SMT(1:60,1) = SMV(iter,1)*ones(60,1);
       SMT(61:124,1) = SMV(iter,2)*ones(64,1);
       SMT(125:157,1) = SMV(iter,3)*ones(33,1);
    end

    SD = SA(1:10:NG,:,:);
    RD = RA(:,1:10:NG);
    QD = QA(:,1:10:NG);
    VD = SV(1:10:NG,:);
    MD = SMV(1:10:NG,:);

    save(DFILE(file,:),varname(1,:),varname(2,:),varname(3,:),varname(4,:),varname(5,:))

    % reinitialize gibbs arrays (buffer for back step)
    SA(1,:) = SA(NG,:);
    QA(:,1) = QA(:,NG);
    RA(:,1) = RA(:,NG);
    SV(1,:) = SV(NG,:);
    SMV(1,:) = SMV(NG,:);

end
exit
