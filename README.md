# Recursive_Stereo_Disill
Unsupervised Monocular Depth Estimation via Recursive Stereo Distillation
## Requirements
This work is implemented using Tensorflow 1.0 and python 2.7.

## Train
```
python MS_main.py --mode=train --data_path=/data3T/KITTI/raw_data/ --filenames_file=./utils/filenames/kitti_train_files.txt --log_directory=models/ --model_name=Test_A --dataset=kitti --encoder=resASPPNet --batch_size=4 --num_epochs=50 --iter_number=2
```
## Test
```
python MS_main.py --mode=train --data_path=/data3T/KITTI/raw_data/ --filenames_file=./utils/filenames/kitti_train_files.txt --log_directory=models/ --model_name=Test_A --dataset=kitti --encoder=resASPPNet --batch_size=4 --num_epochs=50 --iter_number=2
```

We have prepared the testing results [here](链接：https://pan.baidu.com/s/1dygNvYEmTAwStvI6q0o9mw), pwd：gt0E.
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







  

