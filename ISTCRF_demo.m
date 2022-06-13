%%  Iterative Structure Transformation and Conditional Random Field based Method for Unsupervised Multimodal Change Detection
%{
Code: IST-CRF - 2022
This is a test program for the Iterative Structure Transformation and Conditional Random Field based Method for Multimodal Change Detection.

If you use this code for your research, please cite our paper. Thank you!

Sun, Yuli, et al. "Iterative Structure Transformation and Conditional Random Field based Method for Unsupervised Multimodal Change Detection"
Pattern Recognition, 2022,
https://doi.org/10.1016/j.patcog.2022.108845

===================================================
%}

clc
clear;
close all
addpath('auxi_funcs')
%% load dataset
addpath('datasets')
% #2-Img7, #3-Img17, and #5-Img5 can be found at Professor Max Mignotte's webpage (http://www-labs.iro.umontreal.ca/~mignotte/) and they are associated with this paper https://doi.org/10.1109/TGRS.2020.2986239.
% #6-California is download from Dr. Luigi Tommaso Luppino's webpage (https://sites.google.com/view/luppino/data) and it was downsampled to 875*500 as shown in our paper.
% For other datasets, we recommend a similar pre-processing as in "Load_dataset"

% dataset = #1-Italy, #2-Img7, #3-Img17, #4-Shuguang, #5-Img5, #6-California or others
dataset = '#1-Italy';
if strcmp(dataset,'others') == 0
    Load_dataset
elseif strcmp(dataset,'others') == 1
    image_t1 = imread('imaget1.bmp');
    image_t2 = imread('imaget2.bmp');
    gt = imread('Refgt.bmp');
    Ref_gt = double(gt(:,:,1));
    opt.type_t1 = 'sar';% set it to 'optical' for optical image, and 'sar' for SAR image.
    opt.type_t2 = 'optical';% set it to 'optical' for optical image, and 'sar' for SAR image.
    figure;
    subplot(131);imshow(image_t1,[]);title('imaget1')
    subplot(132);imshow(image_t2,[]);title('imaget2')
    subplot(133);imshow(Ref_gt,[]);title('Refgt')
end
Ref_gt = Ref_gt/max(Ref_gt(:));

% When opt.Normalization = 'off', the results are consistent with those in the paper, 
% but opt.Normalization = 'on' usually gives better results

opt.Normalization = 'on'; % 'off' or 'on'; 
if strcmp(opt.Normalization,'on') == 1
    image_t1 = image_normlized(image_t1,opt.type_t1);
    image_t2 = image_normlized(image_t2,opt.type_t2);
end
fprintf(['\n Data loading is completed...... '])
%% Parameter setting
% With different parameter settings, the results will be a little different
% Ns: the number of superpxiels,  A larger Ns will improve the detection granularity, but also increase the running time. 5000 <= Ns <= 10000 is recommended.
% Niter: the maximum number of iterations, 2 <= Niter <=7 is recommended.
% alpha,beta: balance parameters. Vary alpha, beta = [1,3,5,7,9,11,13] and select the best one as the result.

opt.Ns = 5000;
opt.Niter = 7;
opt.alpha = [1:2:13];
opt.beta = [1:2:13];
%% IST-CRF
fprintf(['\n IST-CRF is running...... '])
for i = 1:length(opt.alpha)
    for j = 1:length(opt.beta)  
        fprintf('\n alpha is %4.3f; beta is %4.3f \n',opt.alpha(i),opt.beta(j))
        [CM_map_iter,DIx_iter,DIy_iter] = ISTCRF_main(image_t1,image_t2,opt.Ns,opt.Niter,opt.alpha(i),opt.beta(j));
        for iter = 1:opt.Niter
            [tp,fp,tn,fn,fplv,fnlv,~,~,pcc(iter),kappa(iter),imw]=performance(CM_map_iter(:,:,iter),Ref_gt);
            F1(iter) = 2*tp/(2*tp+fp+fn);
        end
        [~,iter_idx] = max(F1);
        CM_map_parameter(:,:,i,j) = CM_map_iter(:,:,iter_idx);
        DIx_parameter(:,:,i,j) = DIx_iter(:,:,iter_idx);
        DIy_parameter(:,:,i,j) = DIy_iter(:,:,iter_idx);
        pcc_parameter(i,j) = pcc(iter_idx);
        kappa_parameter(i,j) = kappa(iter_idx);
        F1_parameter(i,j) = F1(iter_idx);
    end
end
[idx_alpha, idx_beta] = find(F1_parameter==max(F1_parameter(:)));
CM = CM_map_parameter(:,:,idx_alpha(1),idx_beta(1));
DIx = DIx_parameter(:,:,idx_alpha(1),idx_beta(1));
DIy = DIy_parameter(:,:,idx_alpha(1),idx_beta(1));
pcc = pcc_parameter(idx_alpha(1),idx_beta(1));
kappa = kappa_parameter(idx_alpha(1),idx_beta(1));
F1 = F1_parameter(idx_alpha(1),idx_beta(1));
%% Displaying results

fprintf(['\n Displaying the results...... '])
opt_parameters = '\n The opt.alpha is %4.3f; the opt.beta is %4.3f \n';
fprintf(opt_parameters,opt.alpha(idx_alpha(1)),opt.beta(idx_beta(1)))
result = '\n PCC is %4.3f; kappa is %4.3f; F1 is %4.3f \n';
fprintf(result,pcc,kappa,F1)
figure;
subplot(131);imshow(DIx,[]);title('Difference image')
subplot(132);imshow(DIy,[]);title('Difference image')
subplot(133);imshow(CM,[]);title('Change mape')

if F1 < 0.3
   fprintf('\n');disp('Please select the appropriate opt.alfa and opt.beta for IST-CRF!')
end  
