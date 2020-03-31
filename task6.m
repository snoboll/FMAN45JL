%% task 6
%training
lambda_grid = exp(linspace(log(0.1), log(10), 5));

[Wopt,lambdaopt,RMSEval,RMSEest] = skeleton_multiframe_lasso_cv(Ttrain, Xaudio, lambda_grid, 2);

%% plot
figure(1)
clf;
hold on;
plot(lambda_grid, RMSEest);
plot(lambda_grid, RMSEval);
xline(lambdaopt, '--', ('Optimal lambda'));
xlabel('lambda');
ylabel('RMSE');
set(gca, 'Xscale', 'log');
legend('RMSEest', 'RMSEval');


%% task 7

[Yclean] = lasso_denoise(Ttest, Xaudio, lambdaopt);

save('denoised_audio','Ytest','fs');