function [PairWise_EdgeWeights,LSANedgeWeights,GSSNedgeWeights_x,GSSNedgeWeights_y] = PairwisePotential(sup_img,alpha,beta,t1_feature,t2_feature,Kmat)
[h,w]   = size(sup_img);
nbr_sp  = max(sup_img(:));
idx_co = label2idx(sup_img);
for i = 1:nbr_sp
    index_vector = idx_co{i};
    [location_x location_y] = ind2sub(size(sup_img),index_vector);
    location_center(i,:) = [round(mean(location_x)) round(mean(location_y))];
end
%% R-adjacency neighborhood system
adj_mat = zeros(nbr_sp);
for i=2:h-1
    for j=2:w-1
        label = sup_img(i,j);
        if (label ~= sup_img(i+1,j-1))
            adj_mat(label, sup_img(i+1,j-1)) = 1;
        end
        if (label ~= sup_img(i,j+1))
            adj_mat(label, sup_img(i,j+1)) = 1;
        end
        if (label ~= sup_img(i+1,j))
            adj_mat(label, sup_img(i+1,j)) = 1;
        end
        if (label ~= sup_img(i+1,j+1))
            adj_mat(label, sup_img(i+1,j+1)) = 1;
        end
    end
end
adj_mat_1 = double((adj_mat + adj_mat')>0);
R = 2*round(sqrt(h*w/nbr_sp));
adj_mat = zeros(nbr_sp);
for i=1:nbr_sp
    for j = i:nbr_sp
        if ((location_center(i,1) - location_center(j,1))^2 + (location_center(i,2) - location_center(j,2))^2 < R^2)
            adj_mat (i,j) = 1;
        end
    end
end
adj_mat = double((adj_mat + adj_mat')>0);
adj_mat_2 = adj_mat - eye(nbr_sp);
adj_mat = adj_mat_1|adj_mat_2;
%% PairWise edgeWeights

%  LSAN based Pairwise Potential

LSANedgeWeights = zeros(sum(adj_mat(:)),4);
[node_x node_y] = find(adj_mat ==1);
LSANedgeWeights(:,1) = node_x; % index of node 1
LSANedgeWeights(:,2) = node_y; % index of node 2
for i = 1:sum(adj_mat(:))
    index_node_x = LSANedgeWeights(i,1);
    index_node_y = LSANedgeWeights(i,2);
    feature_t1_x = t1_feature(:,index_node_x);
    feature_t1_y = t1_feature(:,index_node_y);
    feature_t2_x = t2_feature(:,index_node_x);
    feature_t2_y = t2_feature(:,index_node_y);
    Dpq_t1(i) = norm(feature_t1_x-feature_t1_y,2)^2;
    Dpq_t2(i) = norm(feature_t2_x-feature_t2_y,2)^2;
    dist(i) = max(norm(location_center(index_node_x,:)-location_center(index_node_y,:),2),1);
end
sigma_t1 = mean(Dpq_t1);
sigma_t2 = mean(Dpq_t2);
for i =  1:sum(adj_mat(:))
    if Dpq_t1(i) <= sigma_t1 && Dpq_t2(i) <= sigma_t2
        Vpq(i) = exp(-Dpq_t1(i)/(2*sigma_t1)-Dpq_t2(i)/(2*sigma_t2));
    elseif Dpq_t1(i) <= sigma_t1 && Dpq_t2(i) > sigma_t2
        Vpq(i) = exp(Dpq_t1(i)/(2*sigma_t1)-1-Dpq_t2(i)/(2*sigma_t2));
    elseif Dpq_t1(i) > sigma_t1 && Dpq_t2(i) <= sigma_t2
        Vpq(i) = exp(-Dpq_t1(i)/(2*sigma_t1)+Dpq_t2(i)/(2*sigma_t2)-1);
    else
        Vpq(i) = exp(-1);
    end
end
Vpq = Vpq ./dist;
LSANedgeWeights(:,3) = alpha*Vpq;                  % node 1 ---> node 2
LSANedgeWeights(:,4) = alpha*Vpq;                  % node 2 ---> node 1

% GSSN based Pairwise Potential

X = t1_feature';
Y = t2_feature';
kmax = max(Kmat(:))+1;
Kadj_matx = zeros(nbr_sp);
Kadj_maty = zeros(nbr_sp);
[idx_org, ~] = knnsearch(X,X,'k',kmax);
[idy_org, ~] = knnsearch(Y,Y,'k',kmax);
for i = 1: nbr_sp
    kx = Kmat(i,1);
    ky = Kmat(i,3);
    idx_org_k = idx_org(i,2:kx);
    idy_org_k = idy_org(i,2:ky);
    sigma_x(i) = mean(pdist2(X(idx_org_k,:),X(i,:)).^2);
    sigma_y(i) = mean(pdist2(Y(idy_org_k,:),Y(i,:)).^2);
    Kadj_matx(i,idx_org_k) = 1;
    Kadj_maty(i,idy_org_k) = 1;
end
GSSNedgeWeights_x = zeros(sum(Kadj_matx(:)),4);
[Knodex_i Knodex_j] = find(Kadj_matx ==1);
GSSNedgeWeights_x(:,1) = Knodex_i; % index of node 1
GSSNedgeWeights_x(:,2) = Knodex_j; % index of node 2

GSSNedgeWeights_y = zeros(sum(Kadj_maty(:)),4);
[Knodey_i Knodey_j] = find(Kadj_maty ==1);
GSSNedgeWeights_y(:,1) = Knodey_i; % index of node 1
GSSNedgeWeights_y(:,2) = Knodey_j; % index of node 2
for i = 1:sum(Kadj_matx(:)) % KNN of X
    index_node_i = GSSNedgeWeights_x(i,1);
    index_node_j = GSSNedgeWeights_x(i,2);
    feature_t2_i = t2_feature(:,index_node_i);
    feature_t2_j = t2_feature(:,index_node_j);
    feature_t2_distance(i) = (norm(feature_t2_i-feature_t2_j,2)^2)/(sigma_y(index_node_i)+sigma_y(index_node_j));
end
for i = 1:sum(Kadj_maty(:)) % KNN of X
    index_node_i = GSSNedgeWeights_y(i,1);
    index_node_j = GSSNedgeWeights_y(i,2);
    feature_t1_i = t1_feature(:,index_node_i);
    feature_t1_j = t1_feature(:,index_node_j);
    feature_t1_distance(i) = (norm(feature_t1_i-feature_t1_j,2)^2)/(sigma_x(index_node_i)+sigma_x(index_node_j));
end

GSSNedgeWeights_x(:,3) = max(beta*(2*exp(-feature_t2_distance)-2*exp(-0.5)),0);
GSSNedgeWeights_x(:,4) = GSSNedgeWeights_x(:,3);
GSSNedgeWeights_y(:,3) = max(beta*(2*exp(-feature_t1_distance)-2*exp(-0.5)),0);
GSSNedgeWeights_y(:,4) = GSSNedgeWeights_y(:,3);

%% PairWise_EdgeWeights

fused_EdgeWeights = [LSANedgeWeights;GSSNedgeWeights_x;GSSNedgeWeights_y];
PairWise_EdgeWeights = accumarray(fused_EdgeWeights(:,1:2),fused_EdgeWeights(:,3),[],[],[],true);
[i,j,v] = find(PairWise_EdgeWeights);
PairWise_EdgeWeights = [i,j,v];
PairWise_EdgeWeights(:,4) = PairWise_EdgeWeights(:,3);
