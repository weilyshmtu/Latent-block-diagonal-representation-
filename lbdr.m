function [H,Z,J,E]=lbdr(X,Lx,k,alpha,beta)

% min HLxH' + alpha|Z|_k + beta |E|_s
% s.t. H = HZ + E;
%      diag(Z) = 0, Z >= 0, Z = Z'

tol = 1e-7;
maxIter = 1e3;
[d,n] = size(X);
rho = 1.1;
max_mu = 1e30;
mu = 1e-6;

e = ones(n,1);
I_n = eye(n); 
 %% Initializing optimization variables
% intialize
J = zeros(n,n);
Z = J;
E = sparse(d,n);
W = zeros(n,n);
Y1 = zeros(d,n);
Y2 = zeros(n,n);

%% Start main loop
iter = 0;
disp(['computing...' ]);
while iter<maxIter
    iter = iter + 1;
    %% update H
    A = 2*Lx + mu*(I_n - Z)*(I_n - Z)';
    B = mu*(E - Y1/mu)*(I_n - Z)';
    H = B/A;
    
    %% update Z
    A = H'*H + I_n;
    B = H'*(H - E + Y1/mu) + J - Y2/mu;
    Z = A\B;
    
    %% update J
    A = Z + Y2/mu - alpha/mu*(diag(W)*e' - W);
    A = max(0,(A+A')/2);
    J = J-diag(diag(J));
    L_J = diag(J*e) - J;
    %% update W
    [U,D] = eig(L_J);
    D = diag(D);
    [~, ind] = sort(D);    
    W = U(:,ind(1:k))*U(:,ind(1:k))';
    
    %% update E
    B = X - X*Z;
    A = B + Y1/mu;
    
     E = max(0,A - beta/mu)+min(0,A + beta/mu);   % l1-norm
    %E = solve_l1l2(A,beta/mu);                        % l21-norm
    %E = mu*A/(2*beta+mu);                % F-norm
    
    leq1 = B - E;
    leq2 = Z - J;
 
    
    stopC = max([max(max(abs(leq1))),max(max(abs(leq2)))]);
    if iter==1 || mod(iter,50)==0 || stopC<tol
        disp(['iter ' num2str(iter) ',mu=' num2str(mu,'%2.1e') ...
            ',stopALM=' num2str(stopC,'%2.3e')]);
    end
    if stopC<tol 
        break;
    else
        Y1 = Y1 + mu*leq1;
        Y2 = Y2 + mu*leq2;
        mu = min(max_mu,mu*rho);
    end
    
end


function [E] = solve_l1l2(W,lambda)
n = size(W,2);
E = W;
for i=1:n
    E(:,i) = solve_l2(W(:,i),lambda);
end


function [x] = solve_l2(w,lambda)
% min lambda |x|_2 + |x-w|_2^2
nw = norm(w);
if nw>lambda
    x = (nw-lambda)*w/nw;
else
    x = zeros(length(w),1);
end