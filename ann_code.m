function [output_test]=ann_code(Y, X, neuron, iteration, trainRatio,valRatio,testRatio)

%Important Note: Data must be desingned like matrix form with obs_num x features



[obs, col]= size(X);
[obs1, col1]= size(Y);

[trainInd,valInd,testInd] = divideint(obs,trainRatio,valRatio,testRatio);

for i=1:length(trainInd)
    train_Y(i,1:col1)= Y(trainInd(i),:);
    train_X(i,:)= X(trainInd(i),:);
end

for i=1:length(testInd)
    test_tar(i,1:col1)= Y(testInd(i), :);
    test_inp(i,:)= X(testInd(i),:);
end
    
for i=1:length(valInd)
    val_Y(i,1:col1)= Y(valInd(i),:);
    val_X(i, :)= X(valInd(i),:);
end


hiddenLayerSize = [neuron neuron];
%hiddenLayerSize = [neuron];
net = patternnet(hiddenLayerSize);
net.performFcn = 'crossentropy'; % 'mse'; 
net.trainFcn = 'trainbfg'; %'traingdx', 'traingd' ; %'trainscg'; % 'trainlm'; % 'trainscg'; %'trainlm'; %'trainbfg'; %  % Levenberg-Marquardt

%net.performParam.regularization = "";%reg_par;
%net.performParam.normalization = 'none';
net.layers{1}.transferFcn = 'tansig';
net.layers{2}.transferFcn = 'tansig';
%net.layers{3}.transferFcn = 'logsig';
%   Dividing the Data (sarch help files to see other options!)

%net.divideFcn= 'divideind';   %  divide data into three parts with respect to their indices. 
%net.divideParam.trainInd = 1:100;
%net.divideParam.valInd = 101:150;    %[] null   '' random;
%net.divideParam.testInd = 151: 200;


%net.divideFcn= 'dividerand';  %  divide data into three parts with respect to their ratios randomly. 
%net.divideParam.trainRatio = 0.8;
%net.divideParam.valRatio = 0.1;
%net.divideParam.testRatio = 0.1;

net.divideFcn= 'divideint';    % divide data into three parts with respect to their ratios as in dealing a deck of cards. However, these separated data can be used in another training.
net.divideParam.trainRatio = trainRatio;
net.divideParam.valRatio = valRatio;
net.divideParam.testRatio = testRatio;

net.trainParam.lr = 0.5;  % for GD GD

net.trainParam.epochs=iteration;
%net.trainParam.goal=0;
%net.trainParam.max_fail=10;

net.trainParam.min_grad=1e-10;


[net,TR]=train(net,X',Y'); 
x=getwb(net)';
view(net)

outputs = net(train_X');
output_test = net(test_inp');
output_all = net(X');
output_val = net(val_X');


perf = mse(net,train_Y',outputs);
test_mse_perf = mse(net,test_tar',output_test);
val_mse_perf = mse(net,val_Y',output_val);
all_mse_perf= mse(net,Y',output_all);

entropy_train = perform(net,train_Y',outputs);
entropy_test = perform(net,test_tar',output_test);
entropy_all = perform(net,Y',output_all);



%par_fix= (neuron*(col +2)+1);   % the number of paramters

par_fix= net.numWeightElements;

fprintf('mse of training data is %6.4f\n',perf);
fprintf('mse of test data is %6.4f\n',test_mse_perf);
fprintf('mse of val data is %6.4f\n',val_mse_perf);
fprintf('mse of all data is %6.4f\n',all_mse_perf);


fprintf('entropy_train is %6.4f\n',entropy_train);
fprintf('entropy_all is %6.4f\n',entropy_all);
fprintf('entropy_test is %6.4f\n',entropy_test);


fprintf('AIC is %6.4f\n',numel(train_Y)*log(perf)+ 2*par_fix);
fprintf('AICc  %6.4f\n',numel(train_Y)*log(perf)+ 2*par_fix + (2*(par_fix+1)*(par_fix+2)/ (numel(train_Y) - par_fix-2))  );
fprintf('bic  %6.4f\n',numel(train_Y)*log(perf)+ par_fix+ par_fix*log(numel(train_Y)) );

end

