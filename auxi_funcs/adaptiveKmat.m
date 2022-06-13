function [Kmat_x] = adaptiveKmat(X_library,X,kmax,kmin)
X = X';
X_library = X_library';
kmax = kmax+1;
[idx, ~] = knnsearch(X_library,X,'k',kmax);
[N,~] = size(X);
degree_x = tabulate(idx(:));
Kmat_x = degree_x(:,2);
Kmat_x(Kmat_x >= kmax) = kmax;
Kmat_x(Kmat_x <= kmin) = kmin;
if length(Kmat_x)<N
    Kmat_x(length(Kmat_x)+1:N) = kmin;
end
