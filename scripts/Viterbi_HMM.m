function [S, deltaMat, c] = Viterbi_HMM(initPr, tranPr, bMat)
%% Viterbi_HMM() implements the Viterbi algorithm for a HMM. 
% Input:
%   - initPr: initial state probability vector. (pi_1, ..., pi_M)
%   - tranPr: transition probability matrix. a_ij = Pr(S_{t+1}=j | S_t=i)
%   - bMat  : pdf values of observations. bMat(t,j) = b_j(x_t) or
%   equivalents
%

%tic
[T, M] = size(bMat);
deltaMat = zeros(T, M);
Psi = zeros(T, M);
c = zeros(T, 1);
S = zeros(T, 1);

tmp = initPr .* bMat(1, :);
c(1) = sum(tmp);                                    % scale factor
deltaMat(1, :) = tmp / c(1);

for t = 2:T
    tmp = deltaMat(t-1, :) .* tranPr';
    [val, idx] = max(tmp,[], 2);
    Psi(t, :) = idx;
    tmp2 = val' .* bMat(t,:);
    c(t) = sum(tmp2);
    deltaMat(t, :) = tmp2 ./ c(t);                      % scaling
end

[~, idx] = max(deltaMat(T, :));
S(T) = idx;

% state sequence backtracking
for t = (T-1):-1:1
    S(t) = Psi(t+1, S(t+1));
end
%toc

end