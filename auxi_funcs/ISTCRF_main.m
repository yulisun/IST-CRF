function [CM_map_iter,DIx_iter,DIy_iter] = ISTCRF_main(image_t1,image_t2,Ns,Niter,alpha,beta)
image_t1 = double(image_t1);
image_t2 = double(image_t2);
%------------- Preprocessing: Superpixel segmentation and feature extraction---------------%

Compactness = 1;
[sup_img] = SLIC_Cosegmentation_v2(image_t1,image_t2,Ns,Compactness);
[t1_feature,t2_feature] = MSMfeature_extraction(sup_img,image_t1,image_t2) ;% MVE;MSM

%------------- IST-CRF ---------------%

%% PairwisePotential

t1_feature_lib = t1_feature;
t2_feature_lib = t2_feature;
iter = 1;
Kmax =round(size(t1_feature,2).^0.5);
Kmin = round(Kmax/10);
[Kmat_x] = adaptiveKmat(t1_feature,t1_feature,Kmax, Kmin);
[Kmat_y] = adaptiveKmat(t2_feature,t2_feature,Kmax, Kmin);
Kmat = [Kmat_x Kmat_x Kmat_y Kmat_y];
labels = zeros(size(t1_feature,2),1);
[PairWise_EdgeWeights] = PairwisePotential(sup_img,alpha,beta,t1_feature,t2_feature,Kmat);
%% Main iterative framework 

while iter<=(Niter)
    
    %  Calculate Change Vectors
    
    idex_unchange = labels== 0;
    t1_feature_lib = t1_feature(:,idex_unchange);
    t2_feature_lib = t2_feature(:,idex_unchange);
    [Kmat_xuc] = adaptiveKmat(t1_feature_lib,t1_feature,Kmax, Kmin);
    [Kmat_yuc] = adaptiveKmat(t2_feature_lib,t2_feature,Kmax, Kmin);
    Kmat = [Kmat_x Kmat_xuc Kmat_y Kmat_yuc];
    [fx, fy] = CalculateChangeVectors(t1_feature_lib',t1_feature',t2_feature_lib',t2_feature',Kmat);
    fx_dist = sqrt(sum(fx.^2,1));fx_dist = fx_dist';
    fy_dist = sqrt(sum(fy.^2,1));fy_dist = fy_dist';
    fx_dist = remove_outlier(fx_dist);
    fy_dist = remove_outlier(fy_dist);
    
    %  Unary Potential
    
    fcm_options = [2 100 1e-5 0];
    [center_x,U_x,~] = fcm(fx',2,fcm_options);
    [center_y,U_y,~] = fcm(fy',2,fcm_options);
    maxU_x = max(U_x);
    index_x1 = find(U_x(1,:) == maxU_x);
    index_x2 = find(U_x(2,:) == maxU_x);
    if mean(fx_dist(index_x1)) > mean(fx_dist(index_x2))
        CM_fcm_x(index_x1) = 1;
        CM_fcm_x(index_x2) = 0;
        wx_change = U_x(1,:);
        wx_unchange = U_x(2,:);
    else
        CM_fcm_x(index_x1) = 0;
        CM_fcm_x(index_x2) = 1;
        wx_change = U_x(2,:);
        wx_unchange = U_x(1,:);
    end
    maxU_y = max(U_y);
    index_y1 = find(U_y(1,:) == maxU_y);
    index_y2 = find(U_y(2,:) == maxU_y);
    if mean(fy_dist(index_y1)) > mean(fy_dist(index_y2))
        CM_fcm_y(index_y1) = 1;
        CM_fcm_y(index_y2) = 0;
        wy_change = U_y(1,:);
        wy_unchange = U_y(2,:);
    else
        CM_fcm_y(index_y1) = 0;
        CM_fcm_y(index_y2) = 1;
        wy_change = U_y(2,:);
        wy_unchange = U_y(1,:);
    end
    UnaryPotential_termWeights(:,1) = (-log(wx_change)) +  (-log(wy_change));
    UnaryPotential_termWeights(:,2) = (-log(wx_unchange)) +  (-log(wy_unchange));

    %  graph cuts
    addpath('GC');
    [cut, labels] = graphCutMex(UnaryPotential_termWeights, PairWise_EdgeWeights);
    
    %% CM and gray DI calculation
    CM_map = suplabel2DI(sup_img,labels);
    CM_map_iter(:,:,iter) = CM_map;
    DIx_iter (:,:,iter)= suplabel2DI(sup_img,fx_dist);
    DIy_iter (:,:,iter)= suplabel2DI(sup_img,fy_dist);
    iter = iter+1;
end