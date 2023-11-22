%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Implementation of several  basic iterative methods for solving linear
% equations
% Constaining - Jacobi, Gauss-Seidel, SOR, Stationary Gradient, Steepest
% Descent and Conjugate Gradient
% Yutao Chen
% 12/2020
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Example
A = [10, -1, 2, 0;
     -1, 11, -1, 3;
     2, -1, 10, -1;
     0, 3, -1, 8];

b = [6; 25; -11; 15];

maxiter = 100;
tol = 1e-4;
sol_init = zeros(4,1);

figure

sol = Jacobi(A,b,sol_init,maxiter,tol);
subplot(2,3,1)
stem(sol)
title('Jacobi Iteration')

sol = GS(A,b,sol_init,maxiter,tol);
subplot(2,3,2)
stem(sol)
title('Gauss-Seidel Iteration')

sol = SOR(A,b,sol_init,maxiter,tol);
subplot(2,3,3)
stem(sol)
title('SOR')

sol = StationaryGrad(A,b,sol_init,maxiter,tol);
subplot(2,3,4)
stem(sol)
title('Stationary Gradient Method')

sol = SteepestDescent(A,b,sol_init,maxiter,tol);
subplot(2,3,5)
stem(sol)
title('Steepest Descent')

sol = CG(A,b,sol_init,maxiter,tol);
subplot(2,3,6)
stem(sol)
title('Conjugate Gradient')
%% Jacobi Iterarion
% Input:
%   A - Ax = b
%   b - Ax = b
%   x_init - Initial guess of the solution
%   maxiter - Maximum iterations
%   tol - Tolerance
% Output:
%   x_new - Results
function x_new = Jacobi(A,b,x_init,maxiter,tol)
    D = diag(diag(A));
    L = tril(A)-D;
    U = triu(A)-D;
    Cj =  - D^-1 * (L + U);
    bj = D^-1;
    
    x_old = x_init;
    
    normVal = inf;
    cnt = 0;
    converged = 1;
    
    while normVal > tol
        x_new = Cj * x_old + bj * b;
        cnt = cnt + 1;
        if cnt > maxiter
            fprintf('Jacobi iteration failed to converge to the desired tolerance %e\n',tol)
            fprintf('because the maximum number of iterations (%d) was reached.\n',maxiter);
            fprintf('The iterate returned with a residual %.4f\n',normVal);
            fprintf('\n');
            converged = 0;
            break
        end
        normVal = norm(A * x_new - b) / norm(b);
        x_old = x_new;
    end
    if converged
        fprintf('Jacobi iteration converged at iteration %d to a solution with residual %e\n',cnt,normVal);
        fprintf('\n');
    end
end

%% Gauss-Seidel Iteration
% Input:
%   A - Ax = b
%   b - Ax = b
%   x_init - Initial guess of the solution
%   maxiter - Maximum iterations
%   tol - Tolerance
% Output:
%   x_new - Results
function x_new = GS(A,b,x_init,maxiter,tol)
    D = diag(diag(A));
    L = tril(A)-D;
    U = triu(A)-D;
    Cgs =  - (D+L)^-1 * U;
    bgs = (D+L)^-1;
    
    x_old = x_init;
    
    normVal = inf;
    cnt = 0;
    converged = 1;
    
    while normVal > tol
        x_new = Cgs * x_old + bgs * b;
        cnt = cnt + 1;
        if cnt > maxiter
            fprintf('Gauss-Seidel iteration failed to converge to the desired tolerance %e\n',tol)
            fprintf('because the maximum number of iterations (%d) was reached.\n',maxiter);
            fprintf('The iterate returned with a residual %.4f\n',normVal);
            fprintf('\n');
            converged = 0;
            break
        end
        normVal = norm(A * x_new - b) / norm(b);
        x_old = x_new;
    end
    if converged
        fprintf('Gauss-Seidel iteration converged at iteration %d to a solution with residual %e\n',cnt,normVal);
        fprintf('\n');
    end
end

%% SOR with optimal Omega
% Input:
%   A - Ax = b
%   b - Ax = b
%   x_init - Initial guess of the solution
%   maxiter - Maximum iterations
%   tol - Tolerance
% Output:
%   x_new - Results
function x_new = SOR(A,b,x_init,maxiter,tol)
    D = diag(diag(A));
    L = tril(A)-D;
    U = triu(A)-D;
    Cgs =  - (D+L)^-1 * U;
    rho = max(abs(eig(Cgs)));
    omega = 2 / (1 + sqrt(1 - rho));
    bgs = (D+L)^-1;
    
    x_old = x_init;
    
    normVal = inf;
    cnt = 0;
    converged = 1;
    
    while normVal > tol
        x_gs = Cgs * x_old + bgs * b;
        x_new = (1 - omega) .* x_old + omega .* x_gs;
        
        cnt = cnt + 1;
        if cnt > maxiter
            fprintf('SOR failed to converge to the desired tolerance %e\n',tol)
            fprintf('because the maximum number of iterations (%d) was reached.\n',maxiter);
            fprintf('The iterate returned with a residual %.4f\n',normVal);
            fprintf('\n');
            converged = 0;
            break
        end
        normVal = norm(A * x_new - b) / norm(b);
        x_old = x_new;
    end
    if converged
        fprintf('SOR converged at iteration %d to a solution with residual %e\n',cnt,normVal);
        fprintf('\n');
    end
end

%% Stationary Gradient
% Input:
%   A - Ax = b
%   b - Ax = b
%   x_init - Initial guess of the solution
%   maxiter - Maximum iterations
%   tol - Tolerance
% Output:
%   x_new - Results
function x_new = StationaryGrad(A,b,x_init,maxiter,tol)
    normVal = inf;
    cnt = 0;
    
    e = eig(A);
    alpha = 2 / (max(e) + min(e));
    
    x_old = x_init;
    converged = 1;
    
    while normVal > tol
        d = b - A * x_old;
        x_new = x_old + alpha * d;
        cnt = cnt + 1;
        if cnt > maxiter
            fprintf('Stationary Gradient failed to converge to the desired tolerance %e\n',tol)
            fprintf('because the maximum number of iterations (%d) was reached.\n',maxiter);
            fprintf('The iterate returned with a residual %.4f\n',normVal);
            fprintf('\n');
            converged = 0;
            break
        end
        normVal = norm(A * x_new - b) / norm(b);
        x_old = x_new;
    end
    if converged
        fprintf('Stationary Gradient converged at iteration %d to a solution with residual %e\n',cnt,normVal);
        fprintf('\n');
    end
end

%% Steepest Descent
% Input:
%   A - Ax = b
%   b - Ax = b
%   x_init - Initial guess of the solution
%   maxiter - Maximum iterations
%   tol - Tolerance
% Output:
%   x_new - Results
function x_new = SteepestDescent(A,b,x_init,maxiter,tol)
    normVal = inf;
    cnt = 0;
    
    x_old = x_init;
    converged = 1;
    
    while normVal > tol
        r = A * x_old - b;
        alpha = dot(r,r) / dot(r,(A * r));
        x_new = x_old - alpha * r;
        cnt = cnt + 1;
        if cnt > maxiter
            fprintf('Steepest Descent failed to converge to the desired tolerance %e\n',tol)
            fprintf('because the maximum number of iterations (%d) was reached.\n',maxiter);
            fprintf('The iterate returned with a residual %.4f\n',normVal);
            fprintf('\n');
            converged = 0;
            break
        end
        normVal = norm(A * x_new - b) / norm(b);
        x_old = x_new;
    end
    if converged
        fprintf('Steepest Descent converged at iteration %d to a solution with residual %e\n',cnt,normVal);
        fprintf('\n');
    end
end

%% Conjugate Gradient
% Input:
%   A - Ax = b
%   b - Ax = b
%   x_init - Initial guess of the solution
%   maxiter - Maximum iterations
%   tol - Tolerance
% Output:
%   x_new - Results
function x_new = CG(A,b,x_init,maxiter,tol)
    normVal = inf;
    cnt = 0;
    
    x_old = x_init;
    r_old = A * x_old - b;
    d_old = r_old;
    converged = 1;
    
    while normVal > tol
        alpha = -(norm(r_old))^2 / sum(d_old .* (A * d_old));
        x_new = x_old + alpha * d_old;
        r_new = A * x_new - b;
        beta = (norm(r_new))^2 / (norm(r_old))^2;
        d_new = r_new + beta * d_old;
        cnt = cnt + 1;
        if cnt > maxiter
            fprintf('Conjugate Gradient failed to converge to the desired tolerance %e\n',tol)
            fprintf('because the maximum number of iterations (%d) was reached.\n',maxiter);
            fprintf('The iterate returned with a residual %.4f\n',normVal);
            fprintf('\n');
            converged = 0;
            break
        end
        normVal = norm(A * x_new - b) / norm(b);
        x_old = x_new;
        r_old = r_new;
        d_old = d_new;
    end
    if converged
        fprintf('Conjugate Gradient converged at iteration %d to a solution with residual %e\n',cnt,normVal);
        fprintf('\n');
    end
end
