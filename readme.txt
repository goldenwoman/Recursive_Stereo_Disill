python monodepth_main.py --mode train --data_path=/data3T/KITTI/raw_data/ --filenames_file=./utils/filenames/kitti_train_files.txt --log_directory models/ --model_name=bilater  --dataset=kitti --encoder=resASPP --batch_size=4 --num_epochs=50 --iter_number=2 --checkpoint_path=/data3T/zml/experiment/lw-eg-recurren-bilater/models/bilater2/model-287100 --retrain

python monodepth_main.py --mode test --data_path=/data3T/KITTI/raw_data/ --filenames_file=/data3T/zml/experiment/lw-eg-recurrent/utils/filenames/eigen_test_files.txt --dataset=kitti --encoder=resASPP --checkpoint_path=/data3T/zml/experiment/lw-eg-recurrent/models/test/model-100 --output_directory=/data3T/zml/experiment/lw-eg-recurrent/models/test/100/ --iter_number=2 --do_stereo

python utils/evaluate_eigen.py --split=eigen --gt_path=/data3T/KITTI/raw_data/ --max_depth 80 --garg_crop --predicted_disp_path=/data3T_1/zml/experiment/lw-eg-monodepth-master/models/test/600/disparities.npy

注意如果这里的batchsize变了，learning rate也应当进行改变
--batch_size=4 --num_epochs=50 --iter_number=2 --learning_rate=5e-5



KITTI Test:

python monodepth_main.py --mode test --data_path=/data3T/KITTI/kitti_data_flow/2015/ --filenames_file=/data3T/zml/experiment/lw-eg-recur-bila-distill+Nonlocal/utils/filenames/kitti_stereo_2015_test_files.txt --dataset=kitti --encoder=resASPP --checkpoint_path=/data3T/zml/experiment/lw-eg-recur-bila-distill+Nonlocal/models/bilater/model-7500 --output_directory=/data3T/zml/experiment/lw-eg-recur-bila-distill+Nonlocal/models/bilater/7500/ --iter_number=2 --do_stereo

python utils/evaluate_kitti.py --split=kitti --gt_path=/data3T/KITTI/kitti_data_flow/2015/ --predicted_disp_path=/data3T/zml/experiment/lw-eg-recur-bila-distill+Nonlocal/models/bilater/7500/7500KITTI/Non-local-disti.npy


python MS_main.py --mode=train --data_path=/data3T/KITTI/raw_data/ --filenames_file=./utils/filenames/kitti_train_files.txt --log_directory=models/ --model_name=Test_B --dataset=kitti --encoder=resASPPNet --batch_size=4 --num_epochs=50 --iter_number=2