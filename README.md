# IST-CRF
Iterative Structure Transformation and Conditional Random Field based Method for Unsupervised Multimodal Change Detection

## Introduction
MATLAB Code: IST-CRF - 2022

This is a test program for the Iterative Structure Transformation and Conditional Random Field based Method (IST-CRF) for multimodal change detection problem.

IST-CRF first constructs graphs to represent the structures of the images, and transforms the heterogeneous images to the same differential domain by using graph based forward and backward structure transformations. Then, the change vectors are calculated to distinguish the changed and unchanged areas. Finally, in order to classify the change vectors and compute the binary change map, a CRF model is designed to fully explore the spectral-spatial information, which incorporates the change information, local spatially-adjacent neighbor information, and global spectrally-similar neighbor information with a random field framework.

Please refer to the paper for details. You are more than welcome to use the code! 

===================================================

## Available datasets and graphCut algorithm

#6-California is download from Dr. Luigi Tommaso Luppino's webpage (https://sites.google.com/view/luppino/data) and it was downsampled to 875*500 as shown in our paper.

#7-Texas is download from Professor Michele Volpi's webpage at https://sites.google.com/site/michelevolpiresearch/home.

The graphCut algorithm is download from Professor Anton Osokin's webpage at https://github.com/aosokin/graphCutMex_BoykovKolmogorov.

If you use these resources, please cite their relevant papers.

===================================================

## Citation

If you use this code for your research, please cite our paper. Thank you!

@article{SUN2022108845,  
title = {Iterative Structure Transformation and Conditional Random Field based Method for Unsupervised Multimodal Change Detection},  
journal = {Pattern Recognition},  
pages = {108845},  
year = {2022},  
issn = {0031-3203},  
doi = {https://doi.org/10.1016/j.patcog.2022.108845},  
url = {https://www.sciencedirect.com/science/article/pii/S0031320322003260}}  

## Future work

The parameters setting of CRF model and multi-scale fusion strategy, which would improve the detection accuracy, will be further explored.

## Running

Unzip the Zip files (GC) and run the IST-CRF demo file (tested in Matlab 2016a)! 

If you have any queries, please do not hesitate to contact me (sunyuli@mail.ustc.edu.cn).
