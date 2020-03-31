%% task5
%training
lambda_grid = exp(linspace(log(0.1), log(10), 50));

[wopt,lambdaopt,RMSEval,RMSEest] = skeleton_lasso_cv(t, X, lambda_grid, 10);

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

%% plotting task 4 with optimal lambda
w4 = skeleton_lasso_ccd(t, X, lambdaopt);

figure(2)
clf;
hold on;
plot(n, t, 'O');
plot(n, X*w4, 'O');
plot(ninterp, Xinterp*w4);
legend('target', 'lambda = 2.2230 = optlambda', 'interpolation');
xlabel('Time');

