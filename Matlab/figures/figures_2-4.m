% posterior predictive simulation
clear

% Add enclosing folder to path so functions defined there are accessible here
addpath("../")

% catalog of data files
DFILE(1,:) = ['../OctaveResults/swuc_swrp_01.mat'];
DFILE(2,:) = ['../OctaveResults/swuc_swrp_02.mat'];
DFILE(3,:) = ['../OctaveResults/swuc_swrp_03.mat'];
DFILE(4,:) = ['../OctaveResults/swuc_swrp_04.mat'];
DFILE(5,:) = ['../OctaveResults/swuc_swrp_05.mat'];
DFILE(6,:) = ['../OctaveResults/swuc_swrp_06.mat'];
DFILE(7,:) = ['../OctaveResults/swuc_swrp_07.mat'];
DFILE(8,:) = ['../OctaveResults/swuc_swrp_08.mat'];
DFILE(9,:) = ['../OctaveResults/swuc_swrp_09.mat'];
DFILE(10,:) = ['../OctaveResults/swuc_swrp_10.mat'];
DFILE(11,:) = ['../OctaveResults/swuc_swrp_11.mat'];
DFILE(12,:) = ['../OctaveResults/swuc_swrp_12.mat'];
DFILE(13,:) = ['../OctaveResults/swuc_swrp_13.mat'];
DFILE(14,:) = ['../OctaveResults/swuc_swrp_14.mat'];
DFILE(15,:) = ['../OctaveResults/swuc_swrp_15.mat'];
DFILE(16,:) = ['../OctaveResults/swuc_swrp_16.mat'];
DFILE(17,:) = ['../OctaveResults/swuc_swrp_17.mat'];
DFILE(18,:) = ['../OctaveResults/swuc_swrp_18.mat'];
DFILE(19,:) = ['../OctaveResults/swuc_swrp_19.mat'];
DFILE(20,:) = ['../OctaveResults/swuc_swrp_20.mat'];
NF = size(DFILE,1);
NB = 11;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% load data, partition sample
% A = xlsread('C:\paolo2\programs\UK\UKdata','Price Data','c2:c804'); % global financial database
load ../../data/UKdata.txt
A = UKdata(:, 2);
y =(log(A(2:end))-log(A(1:end-1)));
[T,N] = size(y);
date = 1210 + [0:1:T-1]';

% splicing the data
Y0 = y(512:581);  % 1721-1790 training sample
YS_1948_2011 = y(739:802); % 1948-2011
clear A y

load ../../data/Lindert_Williamson.txt -ascii % Lindert-Williamson (B.R. Mitchell, British Historical Statistics)
lnP = log(Lindert_Williamson(:,2));
YS_1791_1850 = diff(lnP);
clear Lindert_Williamson lnP

load ../../data/Bowley.txt -ascii % Bowley (B.R. Mitchell, British Historical Statistics)
lnP = log(Bowley(:,2));
YS_1847_1914 = diff(lnP);
clear Bowley lnP

load ../../data/LaborDepartment.txt -ascii % Labor department (B.R. Mitchell, British Historical Statistics)
lnP = log(LaborDepartment(:,2));
YS_1915_1947 = diff(lnP);
clear LaborDepartment lnP

% spliced data: L-W, B, LD, GFD
y = [YS_1791_1850; YS_1847_1914(5:end); YS_1915_1947; YS_1948_2011];
T = size(y,1);
date = 1791 + [0:1:T-1]';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% standard deviation of innovations to log volatilities
% load VA
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

NMC = size(VA,2);
% bound was used for sensitivity analysis at an early stage and has been
% deactivated
dd = ones(NMC,1); % indicator that equals 1 if the bound is satisfied
% for ii = 1:NMC,
%     if max(VA(:,ii)) > bound,
%         dd(ii,1) = 0;
%     end
% end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% standard deviation of pre-1948 measurement errors
% load MA
load(DFILE(NB,:),'MD')
[P,N] = size(MD);
MA = zeros(N,(NF-NB+1)*P);
MA(:,1:P) = MD';
clear MD
for i = NB+1:NF,
  j = i - NB + 1;
  load(DFILE(i,:),'MD')
  MA(:,(j-1)*P+1:j*P) = MD';
  clear MD;
end

% prior for \sigma_r,\sigma_q
v0 = 10;
svr0 = 0.2236*sqrt((v0+1)/v0); % stock and watson's calibrated value adjusted for time aggregation
dr0 = v0*(svr0^2);
svmc1 = zeros(100000,1);
for ii = 1:100000,
    v = ig2(v0,dr0,0);
    svmc1(ii,1) = v^.5;
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% prior for measurement-error variance \sigma_m (prior is same for all periods)
vm0 = 7;
R0 = .0651^2;
sm0 = 0.5*sqrt(R0)*sqrt((vm0+1)/vm0);
dm0 = vm0*(sm0^2);
svmc2 = zeros(100000,1);
for ii = 1:100000,
    v = ig2(vm0,dm0,0);
    svmc2(ii,1) = v^.5;
end

[Np1,Xp1] = hist(svmc1(:,1),50);
[Nt,Xt] = hist(VA(1,:),Xp1);
[Ns,Xs] = hist(VA(2,:),Xp1);
[Np2,Xp2] = hist(svmc2(:,1),50);
[Nm1,Xm1] = hist(MA(1,:),Xp2);
[Nm2,Xm2] = hist(MA(2,:),Xp2);
[Nm3,Xm3] = hist(MA(3,:),Xp2);

figure
plot(Xp1,Np1/sum(Np1'),'--b','Linewidth',2); hold on;
plot(Xt,Nt/sum(Nt'),'-b','Linewidth',2); hold on;
plot(Xs,Ns/sum(Ns'),'-r','Linewidth',2); hold on;
xlabel('\sigma_r,\sigma_q','Fontsize',18)
%title('Prior and Posteriors for the Standard Deviations of Log-Volatility Innovations','Fontsize',16)
legend('Prior','Posterior \sigma_r','Posterior \sigma_q')
set(gca,'Fontsize',16)

figure
plot(Xp2,Np2/sum(Np2'),'--b','Linewidth',2); hold on;
plot(Xm1,Nm1/sum(Nm1'),'-r','Linewidth',2); hold on;
plot(Xm2,Nm2/sum(Nm2'),'-g','Linewidth',2); hold on;
plot(Xm3,Nm3/sum(Nm3'),'-b','Linewidth',2); hold on;
xlabel('\sigma_{im}','Fontsize',18)
%title('Prior and Posteriors for the Standard Deviation of Measurement Innovations','Fontsize',16)
legend('Prior \sigma_m','Posterior \sigma_m 1791-1850','Posterior \sigma_m 1851-1914','Posterior \sigma_m 1915-1947')
set(gca,'Fontsize',16)

clear VA MA svmc1 svmc2 Xp1 Xp2 Xt Xs Np1 Np2 Nt Ns

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
SRA = sort(RA(2:222,:),2);
CRA = SRA(:,NMC*[.25 .5 .75]); % median and interquatile range for RA
clear RA SRA

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
SQA = sort(QA(2:222,:),2);
CQA = SQA(:,NMC*[.25 .5 .75]); % median and interquatile range for RA
clear QA SQA

% compute SA
load(DFILE(NB,:),'SD')
[P,K,T] = size(SD);
SA1 = zeros((NF-NB+1)*P,T); % \pi
SA2 = zeros((NF-NB+1)*P,T); % \mu
SA1(1:P,:,:) = squeeze(SD(:,1,:));
SA2(1:P,:,:) = squeeze(SD(:,2,:));
clear SD
for i = NB+1:NF,
  j = i - NB + 1;
  load(DFILE(i,:),'SD')
  SA1((j-1)*P+1:j*P,:,:) = squeeze(SD(:,1,:));
  SA2((j-1)*P+1:j*P,:,:) = squeeze(SD(:,2,:));
  clear SD;
end
SSA2 = sort(SA2,1)';
CSA2 = SSA2(:,NMC*[.25 .5 .75]); % median and interquatile range
clear SSA2

SAT = SA1-SA2; % transient component
clear SA1 SA2

SSAT = sort(SAT,1)';
CSAT = SSAT(:,NMC*[.25 .5 .75]); % median and interquatile range for RA
clear SAT SSAT

figure
subplot(2,2,1)
plot(date,CSA2(:,2),'-r','Linewidth',2); hold on;
plot(date,CSA2(:,1),'--b','Linewidth',2); hold on;
plot(date,CSA2(:,3),'--b','Linewidth',2); hold off;
title('\mu_{t}','Fontsize',16)
legend('Median','Interquartile range','Location','Northwest')
set(gca,'Fontsize',14)
axis([1791 2011 -inf inf])

subplot(2,2,2)
plot(date,CSAT(:,2),'-r','Linewidth',2); hold on;
plot(date,CSAT(:,1),'--b','Linewidth',2); hold on;
plot(date,CSAT(:,3),'--b','Linewidth',2); hold off;
title('\pi_{t} - \mu_{t}','Fontsize',16)
set(gca,'Fontsize',14)
axis([1791 2011 -inf inf])

subplot(2,2,3)
plot(date,CQA(:,2).^.5,'-r','Linewidth',2); hold on;
plot(date,CQA(:,1).^.5,'--b','Linewidth',2); hold on;
plot(date,CQA(:,3).^.5,'--b','Linewidth',2); hold off;
title('q_{t}^1/2','Fontsize',16)
set(gca,'Fontsize',14)
axis([1791 2011 0 inf])

subplot(2,2,4)
plot(date,CRA(:,2).^.5,'-r','Linewidth',2); hold on;
plot(date,CRA(:,1).^.5,'--b','Linewidth',2); hold on;
plot(date,CRA(:,3).^.5,'--b','Linewidth',2); hold off;
title('r_{t}^1/2','Fontsize',16)
set(gca,'Fontsize',14)
axis([1791 2011 0 inf])

