function [wopt,lambdaopt,RMSEval,RMSEest] = lasso_cv(t,X,lambdavec,K)
% [wopt,lambdaopt,VMSE,EMSE] = lasso_cv(t,X,lambdavec)
% Calculates the LASSO solution problem and trains the hyperparameter using
% cross-validation.
%
%   Output: 
%   wopt        - mx1 LASSO estimate for optimal lambda
%   lambdaopt   - optimal lambda value
%   MSEval      - vector of validation MSE values for lambdas in grid
%   MSEest      - vector of estimation MSE values for lambdas in grid
%
%   inputs: 
%   y           - nx1 data column vector
%   X           - nxm regression matrix
%   lambdavec   - vector grid of possible hyperparameters
%   K           - number of folds

[N,M] = size(X);
Nlam = length(lambdavec);

% Preallocate
SEval = zeros(K,Nlam);
SEest = zeros(K,Nlam);


% cross-validation indexing
randomind = NaN; % Select random indices for validation and estimation
location = 0; % Index start when moving through the folds
Nval = floor(N/K); % How many samples per fold
hop = Nval; % How many samples to skip when moving to the next fold.


for kfold = 1:K
   
    valind = [location+1:location+Nval]; % Select validation indices
    estind = [1:location location+Nval+1:N]; % Select estimation indices
    assert(isempty(intersect(valind,estind)), "There are overlapping indices in valind and estind!"); % assert empty intersection between valind and estind
    wold = zeros(M,1); % Initialize estimate for warm-starting.
    
    for klam = 1:Nlam
        
        what = skeleton_lasso_ccd(t(estind), X(estind,:), lambdavec(klam), wold); % Calculate LASSO estimate on estimation indices for the current lambda-value.
        
        t_val = X(valind,:) * what;
        t_est = X(estind,:) * what;
        
        SEval(kfold,klam) = sum((t(valind) - t_val).^2); % Calculate validation error for this estimate
        SEest(kfold,klam) = sum((t(estind) - t_est).^2); % Calculate estimation error for this estimate
        
        wold = what; % Set current estimate as old estimate for next lambda-value.
        disp(['Fold: ' num2str(kfold) ', lambda-index: ' num2str(klam)]) % Display current fold and lambda-index.
        
    end
    
    location = location+hop; % Hop to location for next fold.
end


MSEval = mean(SEval,1); % Calculate MSE_val as mean of validation error over the folds.
MSEest = mean(SEest,1); % Calculate MSE_est as mean of estimation error over the folds.

RMSEval = sqrt(MSEval);
RMSEest = sqrt(MSEest);

[val, ind] = min(RMSEval);
lambdaopt = lambdavec(ind); % Select optimal lambda 

wopt = skeleton_lasso_ccd(t, X, lambdaopt); % Calculate LASSO estimate for selected lambda using all data.

end

