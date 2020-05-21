% Create predictions for test data (real evaluation of the model)

YPred = [];
%net=resetState(net);
net = predictAndUpdateState(net,XdataTrain{1});
numTimeStepsTest = numel(XdataTest);

%Pull out one time series and use it for prediction
XTest = XdataTest{4};
YTest = YdataTest{4};

net  = resetState(net);
  
[net,YPred] = predictAndUpdateState(net,XTest,'ExecutionEnvironment','cpu');


% Time to draw pretty pictures 
figure
subplot(2,1,1)
plot(YTest)
hold on
plot(YPred,'.-')
hold off
legend(["Simulated" "LSTM Prediction"])
ylabel("Signal Power (dB)")
title("LSTM Model Predictions")

subplot(2,1,2)
stem(YPred - YTest)
xlabel("Time Step")
ylabel("Error")
title("RMSE = ")

