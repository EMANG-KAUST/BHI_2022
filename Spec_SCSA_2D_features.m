%==========================================================================
%     Spectrogram Features extraction based on 2D SCSA formula  using
% separation of variables for $h = ...$ and $\gamma=1$
%
%   Author: Juan M. Vargas and Mohamed A. Bahloul
%    January 22th, 2022
%==========================================================================

clear all


clc

addpath Function

tic

%% Load dataset 

load('./Data/pwdb_data.mat')

%% Spectrogram generation

fs0=500; % Original frecuency sample

fs1=1000; % Desire frecuency sample 

SNR='no'; % Level of noise (dB)

h=0.05; % Semi-classical constant

gm=3; % Gamma value for 2D-SCSA

fe=1;

sig_n="Radial"; % 1:Radial , 2:Digital, 3: Brachial

wav="PPG"; % 1: BP , 2:PPG
window_s=50
s0=50
s1=99

filen=strcat('./Data/2D-SCSA_Spec',wav,'_',sig_n,'_h=',num2str(h),'_gamma=',num2str(gm),'_SNR=',num2str(SNR),'_wins=',num2str(window_s)) % Full name of the folder

mkdir(filen) % Create folder
%%
for i=1:4374
sig=data.waves.P_Radial{1,i}; % Load Signal
if SNR~='no'
sig=arun(sig,SNR,1);  % Add noise to the signal
end

% img=double(pyrunfile("VG_creation.py", "img",signal=sig,tam_me=window_s));
% ps_vec(:,:,i)=img;

sig=(sig-min(sig))/(max(sig)-min(sig)); % Signal min-max normalization
sig=sig';
t0=0:1/fs0:(length(sig)-1)/fs0;
tend=t0(end);
tf=linspace(0,tend,fs1);
vq1= interp1(t0,sig,tf);  % Umsampling signal
 [r,f,t,ps]=spectrogram(vq1,round(length(vq1)/s0),0,s1,1000,'yaxis'); % Create spectrogram 100 x 100
ps_vec(:,:,i)=ps;
end

%%  SCSA features extraction
figg=[1,300,1478,3000,4374]; % images to plot
k=1;
for s=1:4374
img=ps_vec(:,:,s);
img=img(:,:);

%% = = = = = =   The SCSA 2D Method



[img_scsa,psiy,psix,v1,kapx,kapy,Nx,Ny]=SCSA_2D1D(img,h,fe,gm); % Apply 2D1D-SCSA

%% Save images
% if s==figg(k)
% file_surf_ori=strcat(filen,'/','surface_original_img#',num2str(figg(k)),'.png')
% surf_re=surf(img);
% saveas(surf_re,file_surf_ori)
% 
% file_surf_scsa=strcat(filen,'/','surface_SCSA_img#',num2str(figg(k)),'.png')
% surf_re=surf(img_scsa);
% saveas(surf_re,file_surf_scsa)
% 
% k=k+1
% end

%% Reconstruction performnace  
% ERR =(abs(img-img_scsa))./max(max(img)).*100;
% MSE = mean2((img - img_scsa).^2);
% PSNR = 10*log10(1/MSE);
% fprintf('Obtained results : \n PSNR=%f , MSE=%f \n',PSNR, MSE)

%% = = = = = =   Plot results
% figure
% subplot(1,2,1), surf(img), title('Original');
% subplot(1,2,2), surf(img_scsa), title('Reconstructed')

% figure
% subplot(1,2,1),surf(img)
% title('Original spectrogram')
% subplot(1,2,2),surf(img_scsa)
% title('Recostructed spectrogram')

%% = = = = = = Features extraction

% %%%%%%%%%%%%%%%%%%%% Extraction for all rows %%%%%%%%%%%%%%%%%%%%%%%%%%%

% Stadistical based
mean_rows(:,s)=mean(mean(kapx,2));
std_rows(:,s)=mean(std(kapx,0,2));
median_rows(:,s)=mean(median(kapx,2));

% Invariants
INV1_rows(:,s)=mean(4*h*sum(kapx,2));
INV2_rows(:,s)=mean(((16*h)/3) *sum(kapx.^3,2));
INV3_rows(:,s)=mean(((256*h)/7) *sum(kapx.^7,2));

% First three eigen-values

First_rows(:,s)=mean(kapx(:,1));
Second_rows(:,s)=mean(kapx(:,2));
Third_rows(:,s)=mean(kapx(:,3));

% Number of eigen values
N_row(:,s)=Nx;

% Squared-eigen values 
K1sq_row(:,s)=(mean(kapx(:,1)))^2;
K2sq_row(:,s)=(mean(kapx(:,2)))^2;
K3sq_row(:,s)=(mean(kapx(:,3)))^2;

% Ratio
K1r_row(:,s)=mean(kapx(:,1))/h;
K1m_row(:,s)=median(kapx(:,1))/h;

%%%%%%%%%%%%%%%%%%%% Extraction for all columns %%%%%%%%%%%%%%%%%%%%%%%%%%%

% Stadistical based
mean_columns(:,s)=mean(mean(kapy,2));
std_columns(:,s)=mean(std(kapy,0,2));
median_columns(:,s)=mean(median(kapy,2));

% Invariants
INV1_columns(:,s)=mean(4*h*sum(kapy,2));
INV2_columns(:,s)=mean(((16*h)/3) *sum(kapy.^3,2));
INV3_columns(:,s)=mean(((256*h)/7) *sum(kapy.^7,2));

% First three eigen-values


First_columns(:,s)=mean(kapy(:,1));
Second_columns(:,s)=mean(kapy(:,2));
Third_columns(:,s)=mean(kapy(:,3));

% Number of eigenvalues
N_column(:,s)=Ny;

% Squared-eigenvalues
K1sq_column(:,s)=(mean(kapy(:,1)))^2;
K2sq_column(:,s)=(mean(kapy(:,2)))^2;
K3sq_column(:,s)=(mean(kapy(:,3)))^2;

% Ratio
K1r_column(:,s)=mean(kapy(:,1))/h;
K1m_column(:,s)=median(kapy(:,1))/h;

%%%%%%%%%%%%%%%%%%%% Extraction for sum %%%%%%%%%%%%%%%%%%%%%%%%%%%

kapsum=kapx+kapy;

% Stadistical based
mean_sums(:,s)=mean(mean(kapsum,2));
std_sums(:,s)=mean(std(kapsum,0,2));
median_sums(:,s)=mean(median(kapsum,2));

% Invariants
INV1_sums(:,s)=mean(4*h*sum(kapsum,2));
INV2_sums(:,s)=mean(((16*h)/3) *sum(kapsum.^3,2));
INV3_sums(:,s)=mean(((256*h)/7) *sum(kapsum.^7,2));

% First three eigen-values


First_sums(:,s)=mean(kapsum(:,1));
Second_sums(:,s)=mean(kapsum(:,2));
Third_sums(:,s)=mean(kapsum(:,3));

% Number of eigenvalues
N_sum(:,s)=Nx+Ny;

% Squared-eigenvalues
K1sq_sum(:,s)=(mean(kapsum(:,1)))^2;
K2sq_sum(:,s)=(mean(kapsum(:,2)))^2;
K3sq_sum(:,s)=(mean(kapsum(:,3)))^2;

% Ratio
K1r_sum(:,s)=mean(kapsum(:,1))/h;
K1m_sum(:,s)=median(kapsum(:,1))/h;
end

%% Features Selection

tab=struct2table(data.haemods);
Y=tab.('PWV_cf');

%%%%%%%%%%%%%%%%%%%%%% Features based  on SCSA eigenvalues %%%%%%%%%%%%%%%%%%%

mf_c=[K1sq_column',K2sq_column',K3sq_column',K1r_column',K1m_column'];
mf_r=[K1sq_row',K2sq_row',K3sq_row',K1r_row',K1m_row'];
mf_s=[K1sq_sum',K2sq_sum',K3sq_sum',K1r_sum',K1m_sum'];

%%%%%%%%%%%%%%%%% Features based on SCSA statistical moments %%%%%%%%%%%%
       
ms_c=[mean_columns;std_columns;INV1_columns;INV2_columns;INV3_columns;First_columns]';
ms_r=[mean_rows;std_rows;median_rows;INV1_rows;INV2_rows;INV3_rows;First_rows]';
ms_s=[mean_sums;std_sums;median_sums;INV1_sums;INV2_sums;INV3_sums;First_sums]';

%%%%%%%%%%%%%%%%%%%%%% Features matrix  %%%%%%%%%%%%%%%%%%%

X=cat(2,mf_c,mf_r,mf_s,ms_c,ms_r,ms_s);

%X=cat(2,mf_c,ms_s);
for yuyu=1:size(X,2)
 R_y= corrcoef(X(:,yuyu),Y);
 vec_corre(yuyu)=R_y(1,2);
end
vec_corre(abs(vec_corre)<=0.5)=0;
vec_corre(find(isnan(vec_corre)))=0;
indx=find(vec_corre~=0);
features=X(:,indx);


%% Save all the results



file0=strcat(filen,'/','features_indx.csv')
csvwrite(file0,indx)

file=strcat(filen,'/','features_final.csv')
csvwrite(file,features)



%% Star machine learning algorithm (Multiple linear regression)
[RMSE,R2,Y_pred,Y_test] =pyrunfile("Spec_ML_1D_m.py", ["RMSE","R2","y_pred_lr","y_test"],type_sig=sig_n,type_wav=wav,hscsa=num2str(h),gamma=num2str(gm),snr=SNR,ws=num2str(window_s));

metric=array2table([double(RMSE),double(R2)],VariableNames=["RMSE","R-square"]);

writetable(metric,strcat(filen,'/','metric_vec.csv'));

% res=array2table([double(Y_pred).',double(Y_test).'],VariableNames=["Y pred","Y test"]);
% 
% writetable(res,strcat(filen,'/','result_vec.csv'));
