
from __future__ import absolute_import, division, print_function
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import numpy as np
import argparse
import re
import time
import tensorflow as tf
import tensorflow.contrib.slim as slim
import scipy.misc
import matplotlib.pyplot as plt
# import seaborn as sns
import cv2 as cv

from MS_model import *
from MS_dataloader import *
from average_gradients import *
plt.switch_backend('agg')
parser = argparse.ArgumentParser(description='Monocular Depth Estimation via Recursive Stereo Distillation')

parser.add_argument('--mode',                      type=str,   help='train or test', default='train')
parser.add_argument('--model_name',                type=str,   help='model name', default='MSModel')
parser.add_argument('--encoder',                   type=str,   help='type of encoder, vggASPPNet or resASPPNet', default='vggASPPNet')
parser.add_argument('--dataset',                   type=str,   help='dataset to train on, kitti, or cityscapes', default='kitti')
parser.add_argument('--data_path',                 type=str,   help='path to the data', required=True)
parser.add_argument('--filenames_file',            type=str,   help='path to the filenames text file', required=True)
parser.add_argument('--input_height',              type=int,   help='input height', default=256)
parser.add_argument('--input_width',               type=int,   help='input width', default=512)
parser.add_argument('--batch_size',                type=int,   help='batch size', default=8)
parser.add_argument('--num_epochs',                type=int,   help='number of epochs', default=50)
parser.add_argument('--learning_rate',             type=float, help='initial learning rate', default=1e-4)
parser.add_argument('--lr_loss_weight',            type=float, help='left-right consistency weight', default=1.0)
parser.add_argument('--alpha_image_loss',          type=float, help='weight between SSIM and L1 in the image loss', default=0.85)
parser.add_argument('--disp_gradient_loss_weight', type=float, help='disparity smoothness weigth', default=0.5)
parser.add_argument('--do_stereo',                             help='if set, will test on stereo dataset', action='store_true')
parser.add_argument('--wrap_mode',                 type=str,   help='bilinear sampler wrap mode, edge or border', default='border')
parser.add_argument('--use_deconv',                            help='if set, will use transposed convolutions', action='store_true')
parser.add_argument('--num_gpus',                  type=int,   help='number of GPUs to use for training', default=1)
parser.add_argument('--num_threads',               type=int,   help='number of threads to use for data loading', default=8)
parser.add_argument('--output_directory',          type=str,   help='output directory for test disparities, if empty outputs to checkpoint folder', default='')
parser.add_argument('--log_directory',             type=str,   help='directory to save checkpoints and summaries', default='')
parser.add_argument('--checkpoint_path',           type=str,   help='path to a specific checkpoint to load', default='')
parser.add_argument('--retrain',                               help='if used with checkpoint_path, will restart training from step zero', action='store_true')
parser.add_argument('--full_summary',                          help='if set, will keep more data for each summary. Warning: the file can become very large', action='store_true')
parser.add_argument('--iter_number', type=int, help='the iteration number for the recurrent module', default=1)

args = parser.parse_args()


def count_text_lines(file_path):
    f = open(file_path, 'r')
    lines = f.readlines()
    f.close()
    return len(lines)

def train(params):

    with tf.Graph().as_default(), tf.device('/cpu:0'):

        global_step = tf.Variable(0, trainable=False)
        num_training_samples = count_text_lines(args.filenames_file)

        steps_per_epoch = np.ceil(num_training_samples / params.batch_size).astype(np.int32)
        num_total_steps = params.num_epochs * steps_per_epoch
        start_learning_rate = args.learning_rate

        boundaries = [np.int32((3/5) * num_total_steps), np.int32((4/5) * num_total_steps)]
        values = [args.learning_rate, args.learning_rate / 2, args.learning_rate / 4]
        learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)

        opt_step = tf.train.AdamOptimizer(learning_rate)

        print("Total number of samples: {}".format(num_training_samples))
        print("Total number of steps: {}".format(num_total_steps))

        dataloader = MSDataloader(args.data_path, args.filenames_file, params, args.dataset, args.mode)
        left  = dataloader.left_image_batch
        right = dataloader.right_image_batch
        left_splits  = tf.split(left,  args.num_gpus, 0)
        right_splits = tf.split(right, args.num_gpus, 0)

        tower_grads  = []
        tower_losses = []
        reuse_variables = None
        with tf.variable_scope(tf.get_variable_scope()):
            for i in range(args.num_gpus):
                with tf.device('/gpu:%d' % i):

                    model = MSModel(params, args.mode, left_splits[i], right_splits[i], reuse_variables, i)

                    loss = model.total_loss
                    tower_losses.append(loss)

                    reuse_variables = True

                    # # To obtain the training var for distillation
                    # train_vars = [var for var in tf.trainable_variables()]
                    # train_var_lists = [var for var in tf.trainable_variables() if 'model/res50/' in var.name]
                    # grads = opt_step.compute_gradients(loss,var_list=train_var_lists)

                    grads = opt_step.compute_gradients(loss)
                    tower_grads.append(grads)


        grads = average_gradients(tower_grads)

        apply_gradient_op = opt_step.apply_gradients(grads, global_step=global_step)

        total_loss = tf.reduce_mean(tower_losses)

        tf.summary.scalar('learning_rate', learning_rate, ['model_0'])
        tf.summary.scalar('total_loss', total_loss, ['model_0'])
        summary_op = tf.summary.merge_all('model_0')

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True        
        sess = tf.Session(config=config)

        # SAVER
        summary_writer = tf.summary.FileWriter(args.log_directory + '/' + args.model_name, sess.graph)
        train_saver = tf.train.Saver(max_to_keep=10)

        # # To print params
        # total_num_parameters = 0
        # for variable in tf.trainable_variables():
        #     total_num_parameters += np.array(variable.get_shape().as_list()).prod()
        # print("number of trainable parameters: {}".format(total_num_parameters))
        # train_vars = [var for var in tf.trainable_variables()]
        # print('Trainable variables: ')
        # for var in train_vars:
        #     print(var.name)

        # INIT
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        coordinator = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)

        #
        if args.checkpoint_path != '':
            train_saver.restore(sess, args.checkpoint_path.split(".")[0])

            if args.retrain:
                sess.run(global_step.assign(0))

        #
        start_step = global_step.eval(session=sess)
        start_time = time.time()
        for step in range(start_step, num_total_steps):
            before_op_time = time.time()
            _, loss_value = sess.run([apply_gradient_op, total_loss])


            duration = time.time() - before_op_time
            if step and step % 100 == 0:
                examples_per_sec = params.batch_size / duration
                time_sofar = (time.time() - start_time) / 3600
                training_time_left = (num_total_steps / step - 1.0) * time_sofar
                print_string = 'batch {:>6} | examples/s: {:4.2f} | loss: {:.5f} | time elapsed: {:.2f}h | time left: {:.2f}h'
                print(print_string.format(step, examples_per_sec, loss_value, time_sofar, training_time_left))

                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, global_step=step)
            if step and step % 10000 == 0:
                train_saver.save(sess, args.log_directory + '/' + args.model_name + '/model', global_step=step)

        train_saver.save(sess, args.log_directory + '/' + args.model_name + '/model', global_step=num_total_steps)

def test(params):

    dataloader = MSDataloader(args.data_path, args.filenames_file, params, args.dataset, args.mode)
    left  = dataloader.left_image_batch
    right = dataloader.right_image_batch

    model = MSModel(params, args.mode, left, right)


    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True    
    sess = tf.Session(config=config)


    train_saver = tf.train.Saver()

    # INIT
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    coordinator = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)

    #
    if args.checkpoint_path == '':
        restore_path = tf.train.latest_checkpoint(args.log_directory + '/' + args.model_name)
    else:
        restore_path = args.checkpoint_path.split(".")[0]
    train_saver.restore(sess, restore_path)

    num_test_samples = count_text_lines(args.filenames_file)

    print('now testing {} files'.format(num_test_samples))
    disparities     = np.zeros((num_test_samples, params.height, params.width), dtype=np.float32)
    disparities_pp  = np.zeros((num_test_samples, params.height, params.width), dtype=np.float32)
    disparities_ppp = np.zeros((num_test_samples, params.height, params.width), dtype=np.float32)    
    comp_tower  = []
    comp_offset = 10
    for step in range(num_test_samples):
        st = time.time()
        disp, disp_pp, disp_ppp,outtest = sess.run([model.disp_est, model.disp_est_pp, model.disp_est_ppp,model.out])
        if step==74:
            disp_to_img = scipy.misc.imresize(disp, [375, 1242])
            output_d = os.path.dirname(args.output_directory)
            plt.imsave(os.path.join(output_d, "{}_disp.png".format(step)), disp_to_img, cmap='plasma')
            disp_to_img_pp = scipy.misc.imresize(disp_ppp, [375, 1242])
            plt.imsave(os.path.join(output_d, "{}_disp_pp.png".format(step)), disp_to_img_pp, cmap='plasma')
            print('=============done!!!!')
        # print('=====ok')
        # print(disp.shape)
        output_d = os.path.dirname(args.output_directory)
        # disp_to_img_pp = scipy.misc.imresize(disp_pp, [375, 1242])
        # plt.imsave(os.path.join(output_d, "{}_disp.png".format(step)), disp_to_img_pp, cmap='plasma')

        # ##disparities[step] = disp
        #
        # disp_to_img = scipy.misc.imresize(disp, [375, 1242])
        # # aa = disparities_pp[step]
        # #####################################
        # outtest22 = outtest[0:10,0:10]
        # output_d = os.path.dirname(args.output_directory)
        # sns.heatmap(outtest22, linewidths=0.05,  cmap='rainbow')
        # plt.savefig(os.path.join(output_d, "{}_disp_pp111.png".format(step)))
        # plt.imsave(os.path.join(output_d, "{}_disp_pp.png".format(step)), outtest, cmap='plasma')
        # #####################################
        # plt.imsave(os.path.join(output_d, "{}_disp.png".format(step)), disp_to_img, cmap='plasma')
        #
        # disp_to_img_pp = scipy.misc.imresize(disp_ppp, [375, 1242])
        # plt.imsave(os.path.join(output_d, "{}_disp_pp.png".format(step)), disp_to_img_pp, cmap='plasma')
        if step >= comp_offset:
            comp_tower += time.time() - st,
        disparities[step]       = disp
        disparities_pp[step]    = disp_pp
        disparities_ppp[step]   = disp_ppp
    total_time = sum(comp_tower)

    print('done.')
    print('Total time: ', round(total_time, 2))    
    print('Inferece FPS: ', round((num_test_samples-comp_offset)/total_time, 2))

    print('writing disparities.')
    if args.output_directory == '':
        output_directory = os.path.dirname(args.checkpoint_path)
    else:
        output_directory = args.output_directory

    np.save(output_directory + '/disparities.npy',    disparities)
    np.save(output_directory + '/disparities_pp.npy', disparities_pp)
    np.save(output_directory + '/disparities_ppp.npy', disparities_ppp)

    print('done.')

def main(_):

    params = MSModel_parameters(
        encoder=args.encoder,
        height=args.input_height,
        width=args.input_width,
        batch_size=args.batch_size,
        num_threads=args.num_threads,
        num_epochs=args.num_epochs,
        do_stereo=args.do_stereo,
        wrap_mode=args.wrap_mode,
        use_deconv=args.use_deconv,
        alpha_image_loss=args.alpha_image_loss,
        disp_gradient_loss_weight=args.disp_gradient_loss_weight,
        lr_loss_weight=args.lr_loss_weight,
        full_summary=args.full_summary,
        iter_number=args.iter_number)

    if args.mode == 'train':
        train(params)
    elif args.mode == 'test':
        test(params)

if __name__ == '__main__':
    tf.app.run()
