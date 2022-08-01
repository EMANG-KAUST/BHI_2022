%==========================================================================
% Spectrogram-based cf-PWV estimation using stadistical-based features
%
%   Author: Juan M. Vargas
%   E-mail: juan.vargasgarcia@kaust.edu.sa
%    January 22th, 2022
%==========================================================================

clear all

clc

addpath Function

tic

%% Load dataset 

load('./Data/pwdb_data.mat')

%% Editable variables 

SNR="no"; % Level of noise, can be one of the following values: Free-noise, 20 dB
          % 15 dB, 10 dB.

 % The following variables can be changed but to obtain the results shown in 
 % the paper proposed.


sig_n="Radial"; % Signal location 

wav="PPG"; % Signal type

window_s=50 % Spectrogram Window size
s0=50
s1=99






%% Folder creation

filen=strcat('./Data/2D-Shape_Spec',wav,'_',sig_n,'_SNR=',num2str(SNR),'_wins=',num2str(window_s)) % Full name of the folder

mkdir(filen) % Create folder

%% Spectrogram generation

for i=1:4374
sig=data.waves.PPG_Radial{1,i}; % Load Signal
if SNR~='no'
sig=arun(sig,SNR,1);  % Add noise to the signal
end

% img=double(pyrunfile("VG_creation.py", "img",signal=sig,tam_me=window_s));
% ps_vec(:,:,i)=img;

sig=(sig-min(sig))/(max(sig)-min(sig)); % Signal min-max normalization
sig=sig';
t0=0:1/500:(length(sig)-1)/500;
tend=t0(end);
tf=linspace(0,tend,1000);
vq1= interp1(t0,sig,tf);  % Umsampling signal
 [r,f,t,ps]=spectrogram(vq1,round(length(vq1)/s0),0,s1,1000,'yaxis'); % Create spectrogram 100 x 100
ps_vec(:,:,i)=ps;

% Feature extraction
m2= moment(ps,2,'all');
feature(i,1)=log10(sqrt(m2));

feature(i,2) = mean(moment(ps,3,'all'));
feature(i,3) = log10(moment(ps,4,'all'));
spec_nom=(ps-min(ps))/(max(ps)-min(ps));
feature(i,4)=mean(moment(spec_nom,2,'all'));
feature(i,5)=mean(moment(spec_nom,3,'all'));


end

%% Features selection

tab=struct2table(data.haemods);
Y=tab.('PWV_cf');
for yuyu=1:size(feature,2)
 R_y= corrcoef(feature(:,yuyu),Y);
 vec_corre(yuyu)=R_y(1,2);
end
vec_corre(abs(vec_corre)<=0.5)=0;
vec_corre(find(isnan(vec_corre)))=0;
indx=find(vec_corre~=0);
features=feature(:,indx);


%% Save all the results



file0=strcat(filen,'/','features_indx.csv')
csvwrite(file0,indx)

file=strcat(filen,'/','features_final.csv')
csvwrite(file,features)



%% Star machine learning algorithm (Multiple linear regression)
[RMSE,R2,Y_pred,Y_test,hp_mlp,hp_SVR] =pyrunfile("Spec_Stat_ML_1D_m.py", ["RMSE","R2","y_pred_lr","y_test","hp_mlp","hp_SVR"],type_sig=sig_n,type_wav=wav,snr=SNR,ws=num2str(window_s));

metric=array2table([double(RMSE),double(R2)],VariableNames=["RMSE","R-square"]);

writetable(metric,strcat(filen,'/','metric_vec.csv'));


