function [labels, mu, sigma2, mixtureProb, membershipProbMat] = fitGMM_1d_template(x, K)
% fitGMM_1d implements 1-dim Gaussian Mixture Modeling.

% Iteration params
maxIter = 500;
logL = nan(1, maxIter);
emtol = .01;%1e-9; 

% Define objects
x = x(:);
N = length(x);
labels = nan(N, 1);
mu = nan(1, K);
sigma2 = nan(1, K);
mixtureProb = nan(1, K); 

% Set random initial values
mu = rand(1, K) * max(x);
sigma2 = rand(1, K) * max(x);%[1, 1];
mixtureProb = rand(1, K);
mixtureProb = mixtureProb / sum(mixtureProb);


% EM iteration
for iter = 1:maxIter
    % E-step 
    responsibilities = zeros(N, K);
    for i = 1:K
        responsibilities(:,i) = mixtureProb(i) * mvnpdf(x, mu(i), sigma2(i));
    end
    responsibilities = responsibilities ./ sum(responsibilities, 2);

    % M-step
    Nk = sum(responsibilities, 1);
    mixtureProb = Nk ./ N;
    mixtureProb = mixtureProb(:)';

    mu = sum(responsibilities .* x, 1) ./ Nk;
    mu = mu(:)';


    diff = x - mu;
    for i = 1:K
        sigma2(i) = diff(:,i)' * (diff(:,i) .* responsibilities(:,i)) / Nk(i);
    end
    sigma2 = sigma2(:)';

    logl = 0;
    for i = 1:K
        logl = logl + mixtureProb(i) * mvnpdf(x, mu(i), sigma2(i));
    end
    logL(iter) = sum(log(logl));
    % Terminate if converged
    if iter > 1
        deltaLogL = abs((logL(iter) - logL(iter-1)) /  logL(iter-1));
        if deltaLogL < emtol
            disp(iter);
            disp(logL(iter));
            break;
        end
    end
end

% Determine memberships

membershipProbMat = zeros(N, K);
for i = 1:K
    membershipProbMat(:, i) = mixtureProb(i) .* mvnpdf(x, mu(i), sigma2(i));
end
[~, idx] = sort(membershipProbMat, 2);
labels = idx(:, K);

end
