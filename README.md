# Recursive_Stereo_Disill
This repo implements the training and testing of depth upsampling networks for "Learning Scene Structure Guidance via Cross-Task Knowledge Transfer for Single Depth Super-Resolution" by Baoli Sun, Xinchen Ye, and et al. at DLUT.

## System Requirements
This work is implemented using Tensorflow 1.0, CUDA 10.0, python 2.7.

## Train
```
python MS_main.py --mode=train --data_path=/data3T/KITTI/raw_data/ --filenames_file=./utils/filenames/kitti_train_files.txt --log_directory=models/ --model_name=Test_A --dataset=kitti --encoder=resASPPNet --batch_size=4 --num_epochs=50 --iter_number=2
```
## Citation

If this codebase or our method helps your research, please cite:
```
@article{Ye2021tip,
   author = {Xinchen Ye, Xin Fan, Mingliang Zhang, Wei Zhong, Rui Xu},
   title = {Unsupervised Monocular Depth Estimation via Recursive Stereo Distillation},
   booktitle = {IEEE Trans. Image Processing (TIP)},
   year={2021}, 
   volume={0}, 
   pages={0-0}}
```







  




We have prepared two bash scripts to train our models on KITTI and Cityscapes dataset. After preparing the dataset, please run bash file as following (Take kitti dataset as example): 
```
sh ./bash/bash_train_kitti.sh
```
Please configurate the model and output file path by your preference.

## Evaluation 
We have prepared two bash scripts to evaluate the performance of Kitti and Eigen splits on Kitti dataset. Please change the varaiables in the scripts to run the evaluation. You will get the similar results we have in the paper.
* Example on vggASPP model with KITTI training result.
```
sh ./bash/bash_evaluate_kitti.sh
```
You will recieve the results as shown:
```
now testing 200 files
done.
Total time:  4.48
Inferece FPS:  42.44
writing disparities.
done.
>>> 
>>> Kitti: Native Evaluation
   abs_rel,     sq_rel,        rms,    log_rms,     d1_all,         a1,         a2,         a3
    0.1134,     1.1636,      5.734,      0.201,     27.379,      0.853,      0.945,      0.979
>>> Kitti: Post-Processing Evaluation
   abs_rel,     sq_rel,        rms,    log_rms,     d1_all,         a1,         a2,         a3
    0.1079,     1.0259,      5.464,      0.192,     26.395,      0.857,      0.949,      0.982
>>> Kitti: Edge-Guided Post-Processing Evaluation
   abs_rel,     sq_rel,        rms,    log_rms,     d1_all,         a1,         a2,         a3
    0.1077,     1.0238,      5.387,      0.189,     26.152,      0.860,      0.951,      0.983
```
* We skip first 10 testing files in computing FPS due to the unstability of first few iterations. 

## Pre-built models

We have prepared the built models for references [here](https://drive.google.com/open?id=1njgQyNf4Bk5TEQoXzgN4vs31Texi0sxN).
# Recursive_Stereo_Disill
