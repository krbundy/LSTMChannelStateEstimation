%% Data Generation for Time Series Prediction %%
% This simulation is to generate a large amount of data for predicting the
% state of the channel at future time steps.  It also creates labels for
% the next packet to 



%% TO DO: %%
% (1)   Make sure the labels for the packets are for the NEXT packet, not the
%       same packet, and 
% (2)   then make sure that the data is being saved packetwise
%       with each packet as a cell in a cell array.
% (3)   add awgn 





% SIMULATION PARAMS
SAMPLINGRATE = 50000;

% Max Doppler shift for 
MAX_DOPPLER_SHIFT = 1:10:121;

% Array of Path Delays
pathDelays = [1e-9 5e-9 1e-8 5e-8 1e-7 5e-7 1e-6 5e-6 1e-5 5e-5 1e-4 5e-4 1e-3];

% Channel Modulator
mod= comm.QPSKModulator;

%%DEFINE CHANNELS
% Create a cell array of channels for use in creating simulation data for
% model training.

% Channel objects (from Simulink)
channels = {...
    comm.RayleighChannel('SampleRate',SAMPLINGRATE,'MaximumDopplerShift',4,'PathDelays',[0 1e-9],'AveragePathGains',[0 -9]);
    comm.RayleighChannel('SampleRate',SAMPLINGRATE,'MaximumDopplerShift',4,'PathDelays',[0 5e-9],'AveragePathGains',[0 -9]);
    comm.RayleighChannel('SampleRate',SAMPLINGRATE,'MaximumDopplerShift',4,'PathDelays',[0 1e-8],'AveragePathGains',[0 -9]);
    comm.RayleighChannel('SampleRate',SAMPLINGRATE,'MaximumDopplerShift',4,'PathDelays',[0 5e-8],'AveragePathGains',[0 -9]);
    comm.RayleighChannel('SampleRate',SAMPLINGRATE,'MaximumDopplerShift',4,'PathDelays',[0 1e-7],'AveragePathGains',[0 -9]);
    comm.RayleighChannel('SampleRate',SAMPLINGRATE,'MaximumDopplerShift',4,'PathDelays',[0 5e-7],'AveragePathGains',[0 -9]);
    comm.RayleighChannel('SampleRate',SAMPLINGRATE,'MaximumDopplerShift',4,'PathDelays',[0 1e-6],'AveragePathGains',[0 -9]);
    comm.RayleighChannel('SampleRate',SAMPLINGRATE,'MaximumDopplerShift',4,'PathDelays',[0 5e-6],'AveragePathGains',[0 -9]);
    comm.RayleighChannel('SampleRate',SAMPLINGRATE,'MaximumDopplerShift',4,'PathDelays',[0 1e-5],'AveragePathGains',[0 -9]);
    comm.RayleighChannel('SampleRate',SAMPLINGRATE,'MaximumDopplerShift',4,'PathDelays',[0 5e-5],'AveragePathGains',[0 -9]);
    comm.RayleighChannel('SampleRate',SAMPLINGRATE,'MaximumDopplerShift',4,'PathDelays',[0 1e-4],'AveragePathGains',[0 -9]);
    comm.RayleighChannel('SampleRate',SAMPLINGRATE,'MaximumDopplerShift',4,'PathDelays',[0 5e-4],'AveragePathGains',[0 -9]);
    comm.RayleighChannel('SampleRate',SAMPLINGRATE,'MaximumDopplerShift',4,'PathDelays',[0 1e-3],'AveragePathGains',[0 -9]);
    comm.RayleighChannel('SampleRate',SAMPLINGRATE,'MaximumDopplerShift',4,'PathDelays',[0 1e-9],'AveragePathGains',[0 -8]);
    comm.RayleighChannel('SampleRate',SAMPLINGRATE,'MaximumDopplerShift',4,'PathDelays',[0 5e-9],'AveragePathGains',[0 -8]);
    comm.RayleighChannel('SampleRate',SAMPLINGRATE,'MaximumDopplerShift',4,'PathDelays',[0 1e-8],'AveragePathGains',[0 -8]);
    comm.RayleighChannel('SampleRate',SAMPLINGRATE,'MaximumDopplerShift',4,'PathDelays',[0 5e-8],'AveragePathGains',[0 -8]);
    comm.RayleighChannel('SampleRate',SAMPLINGRATE,'MaximumDopplerShift',4,'PathDelays',[0 1e-7],'AveragePathGains',[0 -8]);
    comm.RayleighChannel('SampleRate',SAMPLINGRATE,'MaximumDopplerShift',4,'PathDelays',[0 5e-7],'AveragePathGains',[0 -8]);
    comm.RayleighChannel('SampleRate',SAMPLINGRATE,'MaximumDopplerShift',4,'PathDelays',[0 1e-6],'AveragePathGains',[0 -8]);
    comm.RayleighChannel('SampleRate',SAMPLINGRATE,'MaximumDopplerShift',4,'PathDelays',[0 5e-6],'AveragePathGains',[0 -8]);
    comm.RayleighChannel('SampleRate',SAMPLINGRATE,'MaximumDopplerShift',4,'PathDelays',[0 1e-5],'AveragePathGains',[0 -8]);
    comm.RayleighChannel('SampleRate',SAMPLINGRATE,'MaximumDopplerShift',4,'PathDelays',[0 5e-5],'AveragePathGains',[0 -8]);
    comm.RayleighChannel('SampleRate',SAMPLINGRATE,'MaximumDopplerShift',4,'PathDelays',[0 1e-4],'AveragePathGains',[0 -8]);
    comm.RayleighChannel('SampleRate',SAMPLINGRATE,'MaximumDopplerShift',4,'PathDelays',[0 5e-4],'AveragePathGains',[0 -8]);
    comm.RayleighChannel('SampleRate',SAMPLINGRATE,'MaximumDopplerShift',4,'PathDelays',[0 1e-3],'AveragePathGains',[0 -8]);
    comm.RayleighChannel('SampleRate',SAMPLINGRATE,'MaximumDopplerShift',4,'PathDelays',[0 1e-9],'AveragePathGains',[0 -7]);
    comm.RayleighChannel('SampleRate',SAMPLINGRATE,'MaximumDopplerShift',4,'PathDelays',[0 5e-9],'AveragePathGains',[0 -7]);
    comm.RayleighChannel('SampleRate',SAMPLINGRATE,'MaximumDopplerShift',4,'PathDelays',[0 1e-8],'AveragePathGains',[0 -7]);
    comm.RayleighChannel('SampleRate',SAMPLINGRATE,'MaximumDopplerShift',4,'PathDelays',[0 5e-8],'AveragePathGains',[0 -7]);
    comm.RayleighChannel('SampleRate',SAMPLINGRATE,'MaximumDopplerShift',4,'PathDelays',[0 1e-7],'AveragePathGains',[0 -7]);
    comm.RayleighChannel('SampleRate',SAMPLINGRATE,'MaximumDopplerShift',4,'PathDelays',[0 5e-7],'AveragePathGains',[0 -7]);
    comm.RayleighChannel('SampleRate',SAMPLINGRATE,'MaximumDopplerShift',4,'PathDelays',[0 1e-6],'AveragePathGains',[0 -7]);
    comm.RayleighChannel('SampleRate',SAMPLINGRATE,'MaximumDopplerShift',4,'PathDelays',[0 5e-6],'AveragePathGains',[0 -7]);
    comm.RayleighChannel('SampleRate',SAMPLINGRATE,'MaximumDopplerShift',4,'PathDelays',[0 1e-5],'AveragePathGains',[0 -7]);
    comm.RayleighChannel('SampleRate',SAMPLINGRATE,'MaximumDopplerShift',4,'PathDelays',[0 5e-5],'AveragePathGains',[0 -7]);
    comm.RayleighChannel('SampleRate',SAMPLINGRATE,'MaximumDopplerShift',4,'PathDelays',[0 1e-4],'AveragePathGains',[0 -7]);
    comm.RayleighChannel('SampleRate',SAMPLINGRATE,'MaximumDopplerShift',4,'PathDelays',[0 5e-4],'AveragePathGains',[0 -7]);
    comm.RayleighChannel('SampleRate',SAMPLINGRATE,'MaximumDopplerShift',4,'PathDelays',[0 1e-3],'AveragePathGains',[0 -7]);
    comm.RayleighChannel('SampleRate',SAMPLINGRATE,'MaximumDopplerShift',4,'PathDelays',[0 1e-9],'AveragePathGains',[0 -9]);
    comm.RayleighChannel('SampleRate',SAMPLINGRATE,'MaximumDopplerShift',4,'PathDelays',[0 5e-9],'AveragePathGains',[0 -10]);
    comm.RayleighChannel('SampleRate',SAMPLINGRATE,'MaximumDopplerShift',4,'PathDelays',[0 1e-8],'AveragePathGains',[0 -10]);
    comm.RayleighChannel('SampleRate',SAMPLINGRATE,'MaximumDopplerShift',4,'PathDelays',[0 5e-8],'AveragePathGains',[0 -10]);
    comm.RayleighChannel('SampleRate',SAMPLINGRATE,'MaximumDopplerShift',4,'PathDelays',[0 1e-7],'AveragePathGains',[0 -10]);
    comm.RayleighChannel('SampleRate',SAMPLINGRATE,'MaximumDopplerShift',4,'PathDelays',[0 5e-7],'AveragePathGains',[0 -10]);
    comm.RayleighChannel('SampleRate',SAMPLINGRATE,'MaximumDopplerShift',4,'PathDelays',[0 1e-6],'AveragePathGains',[0 -10]);
    comm.RayleighChannel('SampleRate',SAMPLINGRATE,'MaximumDopplerShift',4,'PathDelays',[0 5e-6],'AveragePathGains',[0 -10]);
    comm.RayleighChannel('SampleRate',SAMPLINGRATE,'MaximumDopplerShift',4,'PathDelays',[0 1e-5],'AveragePathGains',[0 -10]);
    comm.RayleighChannel('SampleRate',SAMPLINGRATE,'MaximumDopplerShift',4,'PathDelays',[0 5e-5],'AveragePathGains',[0 -10]);
    comm.RayleighChannel('SampleRate',SAMPLINGRATE,'MaximumDopplerShift',4,'PathDelays',[0 1e-4],'AveragePathGains',[0 -10]);
    comm.RayleighChannel('SampleRate',SAMPLINGRATE,'MaximumDopplerShift',4,'PathDelays',[0 5e-4],'AveragePathGains',[0 -10]);
    comm.RayleighChannel('SampleRate',SAMPLINGRATE,'MaximumDopplerShift',4,'PathDelays',[0 1e-3],'AveragePathGains',[0 -10]);
    comm.RayleighChannel('SampleRate',SAMPLINGRATE,'MaximumDopplerShift',5,'PathDelays',[0 1e-9],'AveragePathGains',[0 -9]);
    comm.RayleighChannel('SampleRate',SAMPLINGRATE,'MaximumDopplerShift',5,'PathDelays',[0 5e-9],'AveragePathGains',[0 -9]);
    comm.RayleighChannel('SampleRate',SAMPLINGRATE,'MaximumDopplerShift',5,'PathDelays',[0 1e-8],'AveragePathGains',[0 -9]);
    comm.RayleighChannel('SampleRate',SAMPLINGRATE,'MaximumDopplerShift',5,'PathDelays',[0 5e-8],'AveragePathGains',[0 -9]);
    comm.RayleighChannel('SampleRate',SAMPLINGRATE,'MaximumDopplerShift',5,'PathDelays',[0 1e-7],'AveragePathGains',[0 -9]);
    comm.RayleighChannel('SampleRate',SAMPLINGRATE,'MaximumDopplerShift',5,'PathDelays',[0 5e-7],'AveragePathGains',[0 -9]);
    comm.RayleighChannel('SampleRate',SAMPLINGRATE,'MaximumDopplerShift',5,'PathDelays',[0 1e-6],'AveragePathGains',[0 -9]);
    comm.RayleighChannel('SampleRate',SAMPLINGRATE,'MaximumDopplerShift',5,'PathDelays',[0 5e-6],'AveragePathGains',[0 -9]);
    comm.RayleighChannel('SampleRate',SAMPLINGRATE,'MaximumDopplerShift',5,'PathDelays',[0 1e-5],'AveragePathGains',[0 -9]);
    comm.RayleighChannel('SampleRate',SAMPLINGRATE,'MaximumDopplerShift',5,'PathDelays',[0 5e-5],'AveragePathGains',[0 -9]);
    comm.RayleighChannel('SampleRate',SAMPLINGRATE,'MaximumDopplerShift',5,'PathDelays',[0 1e-4],'AveragePathGains',[0 -9]);
    comm.RayleighChannel('SampleRate',SAMPLINGRATE,'MaximumDopplerShift',5,'PathDelays',[0 5e-4],'AveragePathGains',[0 -9]);
    comm.RayleighChannel('SampleRate',SAMPLINGRATE,'MaximumDopplerShift',5,'PathDelays',[0 1e-3],'AveragePathGains',[0 -9]);
    comm.RayleighChannel('SampleRate',SAMPLINGRATE,'MaximumDopplerShift',5,'PathDelays',[0 1e-9],'AveragePathGains',[0 -8]);
    comm.RayleighChannel('SampleRate',SAMPLINGRATE,'MaximumDopplerShift',5,'PathDelays',[0 5e-9],'AveragePathGains',[0 -8]);
    comm.RayleighChannel('SampleRate',SAMPLINGRATE,'MaximumDopplerShift',5,'PathDelays',[0 1e-8],'AveragePathGains',[0 -8]);
    comm.RayleighChannel('SampleRate',SAMPLINGRATE,'MaximumDopplerShift',5,'PathDelays',[0 5e-8],'AveragePathGains',[0 -8]);
    comm.RayleighChannel('SampleRate',SAMPLINGRATE,'MaximumDopplerShift',5,'PathDelays',[0 1e-7],'AveragePathGains',[0 -8]);
    comm.RayleighChannel('SampleRate',SAMPLINGRATE,'MaximumDopplerShift',5,'PathDelays',[0 5e-7],'AveragePathGains',[0 -8]);
    comm.RayleighChannel('SampleRate',SAMPLINGRATE,'MaximumDopplerShift',5,'PathDelays',[0 1e-6],'AveragePathGains',[0 -8]);
    comm.RayleighChannel('SampleRate',SAMPLINGRATE,'MaximumDopplerShift',5,'PathDelays',[0 5e-6],'AveragePathGains',[0 -8]);
    comm.RayleighChannel('SampleRate',SAMPLINGRATE,'MaximumDopplerShift',5,'PathDelays',[0 1e-5],'AveragePathGains',[0 -8]);
    comm.RayleighChannel('SampleRate',SAMPLINGRATE,'MaximumDopplerShift',5,'PathDelays',[0 5e-5],'AveragePathGains',[0 -8]);
    comm.RayleighChannel('SampleRate',SAMPLINGRATE,'MaximumDopplerShift',5,'PathDelays',[0 1e-4],'AveragePathGains',[0 -8]);
    comm.RayleighChannel('SampleRate',SAMPLINGRATE,'MaximumDopplerShift',5,'PathDelays',[0 5e-4],'AveragePathGains',[0 -8]);
    comm.RayleighChannel('SampleRate',SAMPLINGRATE,'MaximumDopplerShift',5,'PathDelays',[0 1e-3],'AveragePathGains',[0 -8]);
    comm.RayleighChannel('SampleRate',SAMPLINGRATE,'MaximumDopplerShift',5,'PathDelays',[0 1e-9],'AveragePathGains',[0 -7]);
    comm.RayleighChannel('SampleRate',SAMPLINGRATE,'MaximumDopplerShift',5,'PathDelays',[0 5e-9],'AveragePathGains',[0 -7]);
    comm.RayleighChannel('SampleRate',SAMPLINGRATE,'MaximumDopplerShift',5,'PathDelays',[0 1e-8],'AveragePathGains',[0 -7]);
    comm.RayleighChannel('SampleRate',SAMPLINGRATE,'MaximumDopplerShift',5,'PathDelays',[0 5e-8],'AveragePathGains',[0 -7]);
    comm.RayleighChannel('SampleRate',SAMPLINGRATE,'MaximumDopplerShift',5,'PathDelays',[0 1e-7],'AveragePathGains',[0 -7]);
    comm.RayleighChannel('SampleRate',SAMPLINGRATE,'MaximumDopplerShift',5,'PathDelays',[0 5e-7],'AveragePathGains',[0 -7]);
    comm.RayleighChannel('SampleRate',SAMPLINGRATE,'MaximumDopplerShift',5,'PathDelays',[0 1e-6],'AveragePathGains',[0 -7]);
    comm.RayleighChannel('SampleRate',SAMPLINGRATE,'MaximumDopplerShift',5,'PathDelays',[0 5e-6],'AveragePathGains',[0 -7]);
    comm.RayleighChannel('SampleRate',SAMPLINGRATE,'MaximumDopplerShift',5,'PathDelays',[0 1e-5],'AveragePathGains',[0 -7]);
    comm.RayleighChannel('SampleRate',SAMPLINGRATE,'MaximumDopplerShift',5,'PathDelays',[0 5e-5],'AveragePathGains',[0 -7]);
    comm.RayleighChannel('SampleRate',SAMPLINGRATE,'MaximumDopplerShift',5,'PathDelays',[0 1e-4],'AveragePathGains',[0 -7]);
    comm.RayleighChannel('SampleRate',SAMPLINGRATE,'MaximumDopplerShift',5,'PathDelays',[0 5e-4],'AveragePathGains',[0 -7]);
    comm.RayleighChannel('SampleRate',SAMPLINGRATE,'MaximumDopplerShift',5,'PathDelays',[0 1e-3],'AveragePathGains',[0 -7]);
    comm.RayleighChannel('SampleRate',SAMPLINGRATE,'MaximumDopplerShift',5,'PathDelays',[0 1e-9],'AveragePathGains',[0 -9]);
    comm.RayleighChannel('SampleRate',SAMPLINGRATE,'MaximumDopplerShift',5,'PathDelays',[0 5e-9],'AveragePathGains',[0 -10]);
    comm.RayleighChannel('SampleRate',SAMPLINGRATE,'MaximumDopplerShift',5,'PathDelays',[0 1e-8],'AveragePathGains',[0 -10]);
    comm.RayleighChannel('SampleRate',SAMPLINGRATE,'MaximumDopplerShift',5,'PathDelays',[0 5e-8],'AveragePathGains',[0 -10]);
    comm.RayleighChannel('SampleRate',SAMPLINGRATE,'MaximumDopplerShift',5,'PathDelays',[0 1e-7],'AveragePathGains',[0 -10]);
    comm.RayleighChannel('SampleRate',SAMPLINGRATE,'MaximumDopplerShift',5,'PathDelays',[0 5e-7],'AveragePathGains',[0 -10]);
    comm.RayleighChannel('SampleRate',SAMPLINGRATE,'MaximumDopplerShift',5,'PathDelays',[0 1e-6],'AveragePathGains',[0 -10]);
    comm.RayleighChannel('SampleRate',SAMPLINGRATE,'MaximumDopplerShift',5,'PathDelays',[0 5e-6],'AveragePathGains',[0 -10]);
    comm.RayleighChannel('SampleRate',SAMPLINGRATE,'MaximumDopplerShift',5,'PathDelays',[0 1e-5],'AveragePathGains',[0 -10]);
    comm.RayleighChannel('SampleRate',SAMPLINGRATE,'MaximumDopplerShift',5,'PathDelays',[0 5e-5],'AveragePathGains',[0 -10]);
    comm.RayleighChannel('SampleRate',SAMPLINGRATE,'MaximumDopplerShift',5,'PathDelays',[0 1e-4],'AveragePathGains',[0 -10]);
    comm.RayleighChannel('SampleRate',SAMPLINGRATE,'MaximumDopplerShift',5,'PathDelays',[0 5e-4],'AveragePathGains',[0 -10]);
    comm.RayleighChannel('SampleRate',SAMPLINGRATE,'MaximumDopplerShift',5,'PathDelays',[0 1e-3],'AveragePathGains',[0 -10])};

%Number of channels
nChannels = length(channels);

%Number of iterations per channel
NITER = 100;

% Channel Demodulator
demod= comm.QPSKDemodulator;


%% Set up some arrays to hold data
%since the MATLAB training system for LSTM models only uses Cell Arrays for
%R2019b, we will use those.
traininglabels = cell(nChannels,1);
testLabels = cell(nChannels,1);

XdataTrain = cell(nChannels,1);
YdataTrain = cell(nChannels,1);
XdataTest = cell(nChannels,1);
YdataTest = cell(nChannels,1);

errors_per_channel = zeros(1,nChannels);

packetwiseTestData = cell(nChannels,1);
packetwiseTrainingData = cell(nChannels,1);

parfor j = 1:nChannels
    
    %initialize more arrays for data specific to the channel
    
    channel = channels{j};
    
    trainingChannelLabels = zeroes(1, 0.9*NITER);
    testChannelLabels = zeroes(1, 0.1*NITER);
    
    trainingPackets = [];
    testPackets = [];
    
    means=zeros(1,NITER);
    variances =  zeros(1,NITER);
    
    %send NITER packets through the simulated channel
    for i = 1:NITER
    %Simulate one pass through the channel
        tx = randi([0 1],100,1);

        qpsktx = mod(tx);

        qpskrx = channel(qpsktx);
        
        rx = demod(qpskrx);
        
        %calculate the hamming distance between the sent and received
        %packets, and then label the packets
        dist = pdist(tx,rx,'hamming');
        
        

        %calculate the channel gain (in dB)
        signal_power_db = 20*log10(abs(qpskrx));

        %Break the data into training and test (90/10 split)
        if( mod(i,10)==0)
           testPackets = vertcat(testPackets,signal_power_db);
           if(dist>0)
                testChannelLabels(i) = 1;
            end
        else
            trainingPackets = vertcat(testPackets,signal_power_db);
            if(dist>0)
                trainingChannelLabels(i) = 1;
            end
        end
        
        %calculate the mean and variance, and save for later
        means(i) = mean(signal_power_db);
        variances(i) = var(signal_power_db);
        
    end
    
    
    % Standardize the data using mean and variance
    SD = sqrt(sum(variances)) / NITER;
    mu = mean(means);
    
    % standardize data and save packetwise
    testLables{j} = testChannelLables;
    trainingLables{j} = trainingChannelLabels;
    
    
    
    % add standardized data to training set
    XdataTrain{j}=((trainingPackets(1:(end-1)) - mu)/SD)';
    YdataTrain{j}=((trainingPackets(2:end) - mu)/SD)';
    
    
    
    
    
    %compute number of errors for simulation validation and model factors;
    %should not be sparse classification
    
    errors_per_channel(j) = sum(trainingChannelLabels);

    traininglabels{j} = trainingChannelLabels(1:70);
    testLabels{j} = trainingChannelLabels(71:end);
    
end



% Tell the world we are done
disp("Channel Simulation Completed: Data Generated")

disp("Mean Errors per 100 packets:")
disp(mean(errors_per_channel))

%save everything* (some exclusions apply)
save( XdataTrain,"trainingDataX.mat")
save( XdataTest, "testDataInputX.mat")
save(YdataTrain, "trainingDataY.mat")
save(YdataTest, "trainingDataY.mat")
save(trainingLabels, "trainingLabels.mat")
save(testLabels, "testLabels.mat")

% shout it from the rooftops
disp("Data Saved")
