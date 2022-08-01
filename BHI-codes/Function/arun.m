function r=arun(s,SNRdB,L)
    
%     AWGN channel
%     Add AWGN noise to input signal. The function adds AWGN noise vector to signal 's' to generate a resulting signal vector 'r' of specified SNR in dB. It also
%     returns the noise vector 'n' that is added to the signal 's' and the power spectral density N0 of noise added
%     Parameters:
%         s : input/transmitted signal vector
%         SNRdB : desired signal to noise ratio (expressed in dB) for the received signal
%         L : oversampling factor (applicable for waveform simulation) default L = 1.
%     Returns:
%         r : received signal vector (r=s+n)

    gamma = 10^(SNRdB/10); %SNR to linear scale
   
    P=L*sum(abs(s).^2)/size(s,1); %Actual power in the vector
    N0=P/gamma; %Find the noise spectral density
    n = sqrt(N0/2)*rand(size(s,1),size(s,2)); %computed noise
    r = s + n; %received signal
  