% SIMULATION PARAMS
DATAFRAMESIZE=10000;
PACKETBITS=8*120;
SAMPLINGRATE = 50000;


% MODEL & TRAINING HYPERPARAMETERS
EPOCHS = 100;
MINIBATCHSIZE = 100;
ALPHA = 0.0001;%learning rate
HIDDENUNITS_LAYER1 = 256;
HIDDENUNITS_LAYER2 = 128;
HIDDENUNITS_LAYER3 = 64;
NUMFEATURES = 1;
NUMOUTPUT = 1;
SERIESLENGTH = 1e5;

% Max Doppler shift for 
MAX_DOPPLER_SHIFT = 1:10:121;

% Array of Path Delays
pathDelays = [1e-9 5e-9 1e-8 5e-8 1e-7 5e-7 1e-6 5e-6 1e-5 5e-5 1e-4 5e-4 1e-3];

% Channel Modulator
mod= comm.QPSKModulator;

% Channel object (from Simulink)

%channel = comm.RayleighChannel('SampleRate',SAMPLINGRATE,'MaximumDopplerShift',4,'PathDelays',[0 1e-7],'AveragePathGains',[0 -9]);
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
    comm.RayleighChannel('SampleRate',SAMPLINGRATE,'MaximumDopplerShift',4,'PathDelays',[0 1e-3],'AveragePathGains',[0 -9])};


% Channel Demodulator
demod= comm.QPSKDemodulator;

tx = zeros(1,PACKETBITS);

XdataTrain = cell(length(pathDelays),1);
YdataTrain = cell(length(pathDelays),1);
XdataTest = cell(length(pathDelays),1);
YdataTest = cell(length(pathDelays),1);

% Network architecture for basic LSTM model

layers = [...
    sequenceInputLayer(NUMFEATURES)
    lstmLayer(HIDDENUNITS_LAYER1)
    dropoutLayer(0.2)
    lstmLayer(HIDDENUNITS_LAYER2)
    dropoutLayer(0.2)
    FullyConnectedLayer(HIDDENUNITS_LAYER3)
    dropoutLayer(0.2)
    fullyConnectedLayer(NUMOUTPUT)
    regressionLayer];

% training options
options = trainingOptions('adam',...
    'MaxEpochs',EPOCHS,...
    'GradientThreshold',1,...
    'InitialLearnRate',0.005,...
    'LearnRateSchedule','piecewise',...
    'LearnRateDropPeriod',20,...
    'LearnRateDropFactor',0.2,...
    'Verbose',0,...
    'Plots','training-progress');


parfor j = 1:length(pathDelays)
    
    channel = channels{j};
    
    %Simulate one pass through the channel
    msg = randi([0 1],SERIESLENGTH,1);
    tx = 
    qpsktx = mod(tx);

    qpskrx = channel(qpsktx);
    rx = demod(qpskrx);

    %calculate the channel gain (in dB)
    signal_power_db = 20*log10(abs(qpskrx));

   %Break the data into training and test
    XdataTest{j} = signal_power_db((0.9*SERIESLENGTH):end-1)';
    YdataTest{j} = signal_power_db((0.9*SERIESLENGTH)+1:end)';

    % Standardize the data using mean and variance
    SD = std(signal_power_db);
    mu = mean(signal_power_db);
    % add standardized data to training set 
    XdataTrain{j}=((signal_power_db(1:(0.9*SERIESLENGTH)-1) - mu)/SD)';
    YdataTrain{j}=((signal_power_db(2:(0.9*SERIESLENGTH)) - mu)/SD)';
end


% Tell the world we are done
disp("Channel Simulation Completed: Data Generated")

save XdataTrain
save XdataTest
save YdataTrain
save YdataTest

disp("Data Saved")

% train the model
[net, log]= trainNetwork(XdataTrain,YdataTrain,layers,options);

% save a cached version of the current model
netCache = net;
save netCache

