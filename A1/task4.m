% task4
w1 = skeleton_lasso_ccd(t, X, 0.1);
nzw1=nnz(w1)

w2 = skeleton_lasso_ccd(t, X, 5);
nzw2=nnz(w2)

w3 = skeleton_lasso_ccd(t, X, 10);
nzw3=nnz(w3)

figure(1)
clf;
hold on;
plot(n, t, 'O');
plot(n, X*w1, 'O');
plot(ninterp, Xinterp*w1);
legend('target', 'lambda = 0.1', 'interpolation');
xlabel('Time');

figure(2)
clf;
hold on;
plot(n, t, 'O');
plot(n, X*w2, 'O');
plot(ninterp, Xinterp*w2)
legend('target', 'lambda = 5', 'interpolation');
xlabel('Time');

figure(3)
clf;
hold on;
plot(n, t, 'O');
plot(n, X*w3, 'O');
plot(ninterp, Xinterp*w3)
legend('target', 'lambda = 10', 'interpolation');
xlabel('Time');



