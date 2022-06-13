function [fx, fy] = CalculateChangeVectors (X_library,X,Y_library,Y,Kmat)
kmax = max(Kmat(:))+1;
[idx_org, ~] = knnsearch(X,X,'k',kmax);
[idy_org, ~] = knnsearch(Y,Y,'k',kmax);
[idx_uc, ~] = knnsearch(X_library,X,'k',kmax);
[idy_uc, ~] = knnsearch(Y_library,Y,'k',kmax);
[N,Dx] = size(X);
[~,Dy] = size(Y);

fx = zeros(Dx,N);
fy = zeros(Dy,N);
for i = 1:N
    kx_o = Kmat(i,1);
    kx_uc = Kmat(i,2);
    ky_o = Kmat(i,3);
    ky_uc = Kmat(i,4);
    idx_org_k = idx_org(i,2:kx_o);
    idy_org_k = idy_org(i,2:ky_o);
    idx_uc_k = idx_uc(i,2:kx_uc);
    idy_uc_k = idy_uc(i,2:ky_uc);
    dix_org = mean(abs(X(idx_org_k,:)-repmat(X(i,:),[kx_o-1,1])));
    diy_org = mean(abs(Y(idy_org_k,:)-repmat(Y(i,:),[ky_o-1,1])));
    dix_uc = mean(abs(X_library(idy_uc_k,:)-repmat(X(i,:),[ky_uc-1,1])));
    diy_uc = mean(abs(Y_library(idx_uc_k,:)-repmat(Y(i,:),[kx_uc-1,1])));
    fx(:,i) = abs(dix_uc - dix_org);
    fy(:,i) = abs(diy_uc - diy_org);
end


