%==========================================================================
% Spectrogram-based cf-PWV estimation using Energy-Texture based features
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

filen=strcat('./Data/2D-Text_Spec',wav,'_',sig_n,'_SNR=',num2str(SNR),'_wins=',num2str(window_s)) % Full name of the folder

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


[mapz] = laws(ps,5);
c=0;
for ims=1:9
 
feature(i,1+(3*c))=mean2(mapz{ims});
feature(i,2+(3*c))=entropy(mapz{ims});
feature(i,3+(3*c))=std2(mapz{ims});
c=c+1;
end

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
[RMSE,R2,Y_pred,Y_test] =pyrunfile("Spec_Text_ML_1D_m.py", ["RMSE","R2","y_pred_lr","y_test"],type_sig=sig_n,type_wav=wav,snr=SNR,ws=num2str(window_s));

metric=array2table([double(RMSE),double(R2)],VariableNames=["RMSE","R-square"]);

writetable(metric,strcat(filen,'/','metric_vec.csv'));

% res=array2table([double(Y_pred).',double(Y_test).'],VariableNames=["Y pred","Y test"]);
% 
% writetable(res,strcat(filen,'/','result_vec.csv'));
