clc;
clear;
close all;

%% -------------------- LOAD DATA --------------------
data = readtable('solar_data.csv');

time = data.Time;
X = data{:, {'Temperature','Humidity','CloudCover','SolarAngle','Radiation'}};
Y = data.Power;

%% -------------------- RUN SIMULINK MODEL --------------------
open_system('Power_PVarray.slx');    
sim('Power_PVarray.slx');            
%% -------------------- TRAIN TEST SPLIT --------------------
cv = cvpartition(size(X,1),'HoldOut',0.2);
X_train = X(training(cv),:);
Y_train = Y(training(cv),:);
X_test  = X(test(cv),:);
Y_test  = Y(test(cv),:);

time_test = time(test(cv));

%% -------------------- MODELS --------------------
% 1. Linear Regression
mdl1 = fitlm(X_train,Y_train);

% 2. Decision Tree
mdl2 = fitrtree(X_train,Y_train);

% 3. Gradient Boosting 
mdl3 = fitrensemble(X_train,Y_train,'Method','LSBoost');

%% -------------------- PREDICTIONS --------------------
Y_pred1 = predict(mdl1,X_test);
Y_pred2 = predict(mdl2,X_test);
Y_pred3 = predict(mdl3,X_test);

%% -------------------- METRICS FUNCTION --------------------
metrics = @(y,ypred) struct( ...
    'R2', 1 - sum((y-ypred).^2)/sum((y-mean(y)).^2), ...
    'RMSE', sqrt(mean((y-ypred).^2)), ...
    'MAE', mean(abs(y-ypred)));

m1 = metrics(Y_test,Y_pred1);
m2 = metrics(Y_test,Y_pred2);
m3 = metrics(Y_test,Y_pred3);

%% --------------------PREDICTED vs ACTUAL GRAPH --------------------
figure;
plot(time_test,Y_test,'b','LineWidth',2); hold on;
plot(time_test,Y_pred3,'r--','LineWidth',2);
xlabel('Time');
ylabel('Solar Power Output');
title('Predicted vs Actual Solar Power');
legend('Actual Power','Predicted Power');
grid on;

%% -------------------- METRICS TABLE --------------------
Model = {'Linear Regression';'Decision Tree';'Gradient Boosting'};
R2 = [m1.R2; m2.R2; m3.R2];
RMSE = [m1.RMSE; m2.RMSE; m3.RMSE];
MAE = [m1.MAE; m2.MAE; m3.MAE];

results_table = table(Model,R2,RMSE,MAE);
disp('Model Accuracy Results:');
disp(results_table);

%% -------------------- MODEL COMPARISON --------------------
figure;
bar(R2);
set(gca,'XTickLabel',Model);
ylabel('R^2 Score');
title('Model Comparison');
grid on;

%% -------------------- FEATURE IMPORTANCE --------------------
importance = predictorImportance(mdl3);

features = {'Temperature','Humidity','CloudCover','SolarAngle','Radiation'};

figure;
bar(importance);
set(gca,'XTickLabel',features);
ylabel('Importance Score');
title('Feature Importance');
grid on;