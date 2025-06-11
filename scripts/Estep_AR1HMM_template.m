function [c, gammaMat, xiArr] = ...
            Estep_AR1HMM_template(x, x1, initPr, tranPr, phi0, phi1, sigmasq, T, M)
% Estep_AR1HMM
%
% J Noh, 2025/02
% Parameters:
%   x: [T,1]: Cumulative summation over time f+ (post-Jordan decomposition)
%   x1: [T,1]: Lagged time series data matrix from x.
%   initPr: [1,M]: Initial state probabilities
%   tranPr: [M,M]: Random transition probabilities
%   phi0: [1,M]: AR(1) intercept for each state
%   phi1: [1,M]: AR(1) coefficient for each state 
%   sigmasq: [1,M]: AR(1) error variance for each state
%   T: [1,1]: Number of time points
%   M: [1,1]: Number of transition states - 1
% Returns:
%   c: [T,1]: Normalizing scale factor for alpha
%   gammaMat: [T,M]: Marginal distribution of latent variables.
%   xiArr: [T, M]: 

%% Define objects
alphaMat = zeros(T, M);
betaMat = zeros(T, M);
xiArr = zeros(T-1, M, M);

c = zeros(T, 1);              % normalizing scale factor for alpha_t(i)
d = zeros(T, 1);              % normalizing scale factor for beta_t(i)

%% pdf function values, b_i(x_t | x_(t-1))
% No need for a for-loop here, just vectorize that part
mu = phi0 + phi1 .* x1(:);
sigma = sigmasq .^ .5;
bMat = normpdf(x(:), mu, sigma);

%% alpha_t(i) forward equation
alphaMat(1,:) = initPr .* bMat(1, :);
c(1) = sum(alphaMat(1,:));
if c(1) > 0
    alphaMat(1,:) = alphaMat(1,:) ./ c(1);
end
for t = 2:T
     alphaMat(t, :) = alphaMat(t-1, :) * tranPr .* bMat(t,:);
     c(t) = sum(alphaMat(t,:));
     if c(t) > 0
        alphaMat(t,:) = alphaMat(t,:) ./ c(t);
     end
end
%disp(alphaMat(1:10,:))

%% beta_t(j) backward equation
betaMat(T,:) = 1;
d(T) = 1;
for t = (T-1):-1:1
    betaMat(t, :) = betaMat(t+1,:) .* betaMat(t+1,:) * tranPr';
    d(t) = sum(betaMat(t,:));
    if d(t) > 0
        betaMat(t,:) = betaMat(t,:) ./ d(t);
    end
end
%disp(betaMat(1:10,:))

%% define gamma_t(i) 
gammaMat = alphaMat .* betaMat;
gammaMat = gammaMat ./ sum(gammaMat, 2); % P(x) = Sum_M alpha(i)beta(i)


%% define xi_t(i,j)  
%bM = bMat(2:T, :) .* betaMat(2:T, :);
for t = 1:T-1
    xiArr(t, :, :) = alphaMat(t,:)' * (bMat(t+1, :) .* betaMat(t+1, :)) .* tranPr;
    %xiArr(t,:,:) = alphaMat(t,:)' * bM(t, :) .* tranPr;
end
xiArr = xiArr ./ sum(xiArr, [2,3]);

end
