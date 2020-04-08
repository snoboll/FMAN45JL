%% task 6
%training 
lambda_grid = exp(linspace (log(0.001), log(1), 50)) %last param = number of lambdas

[Wopt,lambdaopt_audio,RMSEval,RMSEest] = skeleton_multiframe_lasso_cv(Ttrain, Xaudio, lambda_grid, 10); % last param = number of folds

%% plot
figure(1)
clf;
hold on;
plot(lambda_grid, RMSEest);
plot(lambda_grid, RMSEval);
xline(lambdaopt_audio, '--', ('Optimal lambda'));
xlabel('lambda');
ylabel('RMSE');
set(gca, 'Xscale', 'log');
legend('RMSEest', 'RMSEval');


%% task 7

[Yclean] = lasso_denoise(Ttest, Xaudio, 0.1);

save('denoised_audio','Yclean','fs');