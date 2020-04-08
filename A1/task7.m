%% task 7
[Yclean] = lasso_denoise(Ttest, Xaudio, lambdaopt);

%% save data
save('denoised_audio','Ytest','fs');