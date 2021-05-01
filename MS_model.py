
from __future__ import absolute_import, division, print_function
from collections import namedtuple

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

from bilinear_sampler import *

MSModel_parameters = namedtuple('parameters',
                        'encoder, '
                        'height, width, '
                        'batch_size, '
                        'num_threads, '
                        'num_epochs, '
                        'do_stereo, '
                        'wrap_mode, '
                        'use_deconv, '
                        'alpha_image_loss, '
                        'disp_gradient_loss_weight, '
                        'lr_loss_weight, '
                        'iter_number')

class MSModel(object):
    """MSModel"""

    def __init__(self, params, mode, left, right, reuse_variables=None, model_index=0):
        self.params = params
        self.mode = mode
        self.left = left
        self.right = right
        self.model_collection = ['model_' + str(model_index)]

        self.reuse_variables = reuse_variables
        if self.mode == 'test':
            self.img_init()
        self.build_model()
        self.build_outputs()
        if self.mode == 'test':
            return
        self.build_losses()


    def img_init(self, reserve_rate=0.05, reserve_rate_2=0.01):
        b = self.left.get_shape().as_list()[0]
        h = self.left.get_shape().as_list()[1]
        w = self.left.get_shape().as_list()[2]
        l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))

        self.l_mask = 1.0 - np.clip(20 * (l - reserve_rate), 0, 1)
        self.r_mask = np.fliplr(self.l_mask)
        self.m_mask = (1.0 - self.l_mask - self.r_mask)    

        self.l_mask_2 = 1.0 - np.clip(20 * (l - reserve_rate_2), 0, 1)
        self.r_mask_2 = np.fliplr(self.l_mask_2)
        self.m_mask_2 = (1.0 - self.l_mask_2 - self.r_mask_2)


    def img_edge(self, disp, kernel_size=21):
        p = np.floor((kernel_size - 1) / 2).astype(np.int32)
        disp_p = tf.pad(disp, [[0, 0], [0, 0], [p, p], [0, 0]])
        disp_p_ap = tf.nn.avg_pool(disp_p, ksize=[1, 1, p, 1], strides=[1, 1, 1, 1], padding='SAME')
        gx = disp_p_ap[:,:,:(-2*p),:] - disp_p_ap[:,:,(p+1):(-p+1),:]

        gx_max, gx_min = tf.reduce_max(gx), tf.reduce_min(gx)
        return tf.sigmoid((((gx - gx_min) / ( gx_max -  gx_min))-0.5)*32)

    # Get the final stereo disp map
    def Final_disp_S(self, disp):
        l_disp = disp[0,:,:,0]
        r_disp = tf.image.flip_left_right(disp[1,:,:,:])[:,:,0]
        return self.r_mask * l_disp + self.l_mask * r_disp + self.m_mask * l_disp

    # Get the final mono disp map
    def Final_disp_M(self, disp):
        gx = self.img_edge(disp)
        l_disp = disp[0,:,:,0]
        r_disp = tf.image.flip_left_right(disp[1,:,:,:])[:,:,0]
        l_gx = gx[0,:,:,0]
        r_gx = tf.image.flip_left_right(gx[1,:,:,:])[:,:,0]
        w_gx = tf.add(l_gx, r_gx)
        l_gx = tf.truediv(l_gx, w_gx)
        r_gx = tf.truediv(r_gx, w_gx)
        m_disp = l_disp * l_gx + r_disp * r_gx
        return self.r_mask_2 * l_disp + self.l_mask_2 * r_disp + self.m_mask_2 * m_disp

    def gradient_x(self, img):
        gx = img[:,:,:-1,:] - img[:,:,1:,:]
        return gx

    def gradient_y(self, img):
        gy = img[:,:-1,:,:] - img[:,1:,:,:]
        return gy

    def upsample_nn(self, x, ratio):
        h = x.get_shape()[1].value
        w = x.get_shape()[2].value
        return tf.image.resize_nearest_neighbor(x, [h * ratio, w * ratio])

    def scale_pyramid(self, img, num_scales):
        scaled_imgs = [img]
        s = tf.shape(img)
        h = s[1]
        w = s[2]
        for i in range(num_scales - 1):
            ratio = 2 ** (i + 1)
            nh = h // ratio
            nw = w // ratio
            scaled_imgs.append(tf.image.resize_area(img, [nh, nw]))
        return scaled_imgs

    def generate_image_left(self, img, disp):
        return bilinear_sampler_1d_h(img, -disp)

    def generate_image_right(self, img, disp):
        return bilinear_sampler_1d_h(img, disp)

    def SSIM(self, x, y):
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        mu_x = slim.avg_pool2d(x, 3, 1, 'VALID')
        mu_y = slim.avg_pool2d(y, 3, 1, 'VALID')

        sigma_x  = slim.avg_pool2d(x ** 2, 3, 1, 'VALID') - mu_x ** 2
        sigma_y  = slim.avg_pool2d(y ** 2, 3, 1, 'VALID') - mu_y ** 2
        sigma_xy = slim.avg_pool2d(x * y , 3, 1, 'VALID') - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

        SSIM = SSIM_n / SSIM_d

        return tf.clip_by_value((1 - SSIM) / 2, 0, 1)

    def get_smoothness(self, disp, pyramid):
        disp_gradients_x = [self.gradient_x(d) for d in disp]
        disp_gradients_xx = [self.gradient_x(d) for d in disp_gradients_x]
        disp_gradients_y = [self.gradient_y(d) for d in disp]
        disp_gradients_yy = [self.gradient_y(d) for d in disp_gradients_y]

        image_gradients_x = [self.gradient_x(img) for img in pyramid]
        image_gradients_xx = [self.gradient_x(img) for img in image_gradients_x]
        image_gradients_y = [self.gradient_y(img) for img in pyramid]
        image_gradients_yy = [self.gradient_y(img) for img in image_gradients_y]

        weights_x = [tf.exp(-tf.reduce_mean(tf.abs(g), 3, keep_dims=True)) for g in image_gradients_xx]
        weights_y = [tf.exp(-tf.reduce_mean(tf.abs(g), 3, keep_dims=True)) for g in image_gradients_yy]

        smoothness_xx = [disp_gradients_xx[i] * weights_x[i] for i in range(4)]
        smoothness_yy = [disp_gradients_yy[i] * weights_y[i] for i in range(4)]

        return smoothness_xx + smoothness_yy

    def get_disp(self, x):
        disp = 0.3 * self.conv(x, 2, 3, 1, tf.nn.sigmoid)
        return disp

    def conv(self, x, num_out_layers, kernel_size, stride, activation_fn=tf.nn.elu):
        p = np.floor((kernel_size - 1) / 2).astype(np.int32)
        p_x = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]])
        return slim.conv2d(p_x, num_out_layers, kernel_size, stride, 'VALID', activation_fn=activation_fn)

    def conv_block(self, x, num_out_layers, kernel_size):
        conv1 = self.conv(x,     num_out_layers, kernel_size, 1)
        conv2 = self.conv(conv1, num_out_layers, kernel_size, 2)
        return conv2

    def maxpool(self, x, kernel_size):
        p = np.floor((kernel_size - 1) / 2).astype(np.int32)
        p_x = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]])
        return slim.max_pool2d(p_x, kernel_size)

    def resconv(self, x, num_layers, stride):
        do_proj = tf.shape(x)[3] != num_layers or stride == 2
        shortcut = []
        conv1 = self.conv(x,         num_layers, 1, 1)
        conv2 = self.conv(conv1,     num_layers, 3, stride)
        conv3 = self.conv(conv2, 4 * num_layers, 1, 1, None)
        if do_proj:
            shortcut = self.conv(x, 4 * num_layers, 1, stride, None)
        else:
            shortcut = x
        return tf.nn.elu(conv3 + shortcut)

    def resblock(self, x, num_layers, num_blocks):
        out = x
        for i in range(num_blocks - 1):
            out = self.resconv(out, num_layers, 1)
        out = self.resconv(out, num_layers, 2)
        return out

    def upconv(self, x, num_out_layers, kernel_size, scale):
        upsample = self.upsample_nn(x, scale)
        conv = self.conv(upsample, num_out_layers, kernel_size, 1)
        return conv

    def deconv(self, x, num_out_layers, kernel_size, scale):
        p_x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]])
        conv = slim.conv2d_transpose(p_x, num_out_layers, kernel_size, scale, 'SAME')
        return conv[:,3:-1,3:-1,:]

    def NonLocalBlock(self, input_x, height, width,scope='NonLocalBlock'):
        batchsize, wheight, wwidth, in_channels = input_x.get_shape().as_list()
        out_channels = int(in_channels/2)
        with tf.variable_scope(scope) as sc:

            g_x = tf.reshape(input_x, [batchsize, in_channels, -1])
            g_x = tf.transpose(g_x, [0, 2, 1])
            theta_x = tf.reshape(input_x, [batchsize, in_channels, -1])
            theta_xT = tf.transpose(theta_x, [0, 2, 1])
            f = tf.matmul(theta_xT, theta_x)
            f_softmax = tf.nn.softmax(f, -1)
            y = tf.matmul(f_softmax, g_x)
            y = tf.reshape(y, [batchsize, wheight, wwidth, in_channels])
            z = input_x + y*0.01
            return z, f_softmax

    def NonLocalBlockRec(self, input_x, height, width,scope='nonlocal_re'):
        batchsize, wheight, wwidth, in_channels = input_x.get_shape().as_list()
        out_channels = int(in_channels/2)
        with tf.variable_scope(scope) as sca:

            g_x = tf.reshape(input_x, [batchsize, in_channels, -1])
            g_x = tf.transpose(g_x, [0, 2, 1])
            theta_x = tf.reshape(input_x, [batchsize, in_channels, -1])
            theta_xT = tf.transpose(theta_x, [0, 2, 1])
            f = tf.matmul(theta_xT, theta_x)
            f_softmax = tf.nn.softmax(f, -1)
            return f_softmax

    def ASPPModule(self, inputs, depth=256):

        feature_map_size = tf.shape(inputs)
        image_features = tf.reduce_mean(inputs, [1, 2], keep_dims=True)
        image_features = slim.conv2d(image_features, depth, [1, 1], activation_fn=None)
        image_features = tf.image.resize_bilinear(image_features, (feature_map_size[1], feature_map_size[2]))
        atrous_pool_block_1 = slim.conv2d(inputs, depth, [1, 1], activation_fn=None)
        atrous_pool_block_6 = slim.conv2d(inputs, depth, [3, 3], rate=6, activation_fn=None)
        atrous_pool_block_12 = slim.conv2d(inputs, depth, [3, 3], rate=12, activation_fn=None)
        atrous_pool_block_18 = slim.conv2d(inputs, depth, [3, 3], rate=18, activation_fn=None)

        asppmodel = tf.concat((image_features, atrous_pool_block_1, atrous_pool_block_6, atrous_pool_block_12, atrous_pool_block_18), axis=3)

        return asppmodel

    def Refindisp(self, inputD,  inputF):
        conv = self.conv

        with tf.variable_scope('refinedisp'):
            self.bikernel_size = 3
            x = tf.concat([inputD, inputF], 3)
            convr_3 = conv(x, 64, 3, 1)
            convr_4 = conv(convr_3, 64, 3, 1)
            convr_5 = conv(convr_4, 32, 3, 1)
            convr_6 = conv(convr_5, 32, 3, 1)
            convr_7 = conv(convr_6, 2*self.bikernel_size * self.bikernel_size, 3, 1)
            feat_kernel = tf.nn.softmax(-convr_7 ** 2, axis=3)
        return feat_kernel

    def Refindisp_ex(self, inputD):
        return tf.extract_image_patches(images=inputD, ksizes=[1,self.bikernel_size,self.bikernel_size,1], strides=[1, 1, 1, 1], rates=[1, 1, 1, 1],
                                     padding='SAME',name="test")

    # Networks

    def build_vggASPPNet(self):

        conv = self.conv
        if self.params.use_deconv:
            upconv = self.deconv
        else:
            upconv = self.upconv

        with tf.variable_scope('encoder'):
            conv1 = self.conv_block(self.model_input,  32, 7)
            conv2 = self.conv_block(conv1,             64, 5)
            conv3 = self.conv_block(conv2,            128, 3)
            conv4 = self.conv_block(conv3,            256, 3)
            pool4 = self.maxpool(conv4,                    3)
            conv5 = self.ASPPModule(pool4)

        with tf.variable_scope('skips'):          
            skip1 = conv1
            skip2 = conv2
            skip3 = conv3
            skip4 = conv4
        
        # DECODING
        with tf.variable_scope('decoder'):
            upconv5 = upconv(conv5, 256, 3, 2)
            concat5 = tf.concat([upconv5, skip4], 3)
            iconv5  = conv(concat5,  256, 3, 1)

            upconv4 = upconv(iconv5, 128, 3, 2)
            concat4 = tf.concat([upconv4, skip3], 3)
            iconv4  = conv(concat4,  128, 3, 1)
            self.disp4 = self.get_disp(iconv4)
            udisp4  = self.upsample_nn(self.disp4, 2)

            upconv3 = upconv(iconv4,  64, 3, 2)
            concat3 = tf.concat([upconv3, skip2, udisp4], 3)
            iconv3  = conv(concat3,   64, 3, 1)
            self.disp3 = self.get_disp(iconv3)
            udisp3  = self.upsample_nn(self.disp3, 2)

            upconv2 = upconv(iconv3,  32, 3, 2)
            concat2 = tf.concat([upconv2, skip1, udisp3], 3)
            iconv2  = conv(concat2,   32, 3, 1)
            self.disp2 = self.get_disp(iconv2)
            udisp2  = self.upsample_nn(self.disp2, 2)

            upconv1 = upconv(iconv2,  16, 3, 2)
            concat1 = tf.concat([upconv1, udisp2], 3)
            iconv1  = conv(concat1,   16, 3, 1)
            self.disp1 = self.get_disp(iconv1)


    def build_resASPPNet(self):

        conv   = self.conv
        if self.params.use_deconv:
            upconv = self.deconv
        else:
            upconv = self.upconv

        with tf.variable_scope('encoder'):
            conv1 = conv(self.model_input, 64, 7, 2)
            pool1 = self.maxpool(conv1,           3)
            conv2 = self.resblock(pool1,      64, 3)
            conv3 = self.resblock(conv2,     128, 4)
            pool3 = self.maxpool(conv3,           3)
            conv4 = self.ASPPModule(pool3)


        with tf.variable_scope('skips'):     
            skip1 = conv1
            skip2 = pool1
            skip3 = conv2
            skip4 = conv3
        
        # DECODING
        with tf.variable_scope('decoder'):
            upconv5 = upconv(conv4, 256, 3, 2)
            # upconv5, self.softmap5 = self.NonLocalBlock(upconv5,16,32, scope='nonlocal_block2')

            concat5 = tf.concat([upconv5, skip4], 3)
            self.iconv5  = conv(concat5,  256, 3, 1)

            upconv4 = upconv(self.iconv5, 128, 3, 2)
            # upconv4, self.softmap4 = self.NonLocalBlock(upconv4,16,32, scope='nonlocal_block3')
            concat4 = tf.concat([upconv4, skip3], 3)
            self.iconv4  = conv(concat4,  128, 3, 1)
            self.disp4 = self.get_disp(self.iconv4)
            udisp4  = self.upsample_nn(self.disp4, 2)

            upconv3 = upconv(self.iconv4,  64, 3, 2)
            # upconv3, self.softmap3 = self.NonLocalBlock(upconv3, 16, 32, scope='nonlocal_block4')
            concat3 = tf.concat([upconv3, skip2, udisp4], 3)
            self.iconv3  = conv(concat3,   64, 3, 1)
            self.disp3 = self.get_disp(self.iconv3)
            udisp3  = self.upsample_nn(self.disp3, 2)

            upconv2 = upconv(self.iconv3,  32, 3, 2)
            concat2 = tf.concat([upconv2, skip1, udisp3], 3)
            self.iconv2  = conv(concat2,   32, 3, 1)
            self.disp2 = self.get_disp(self.iconv2)
            udisp2  = self.upsample_nn(self.disp2, 2)

            upconv1 = upconv(self.iconv2,  16, 3, 2)
            concat1 = tf.concat([upconv1, udisp2], 3)
            self.iconv1  = conv(concat1,   16, 3, 1)
            self.disp1 = self.get_disp(self.iconv1)

        return self.disp1, self.disp2, self.disp3, self.disp4, self.iconv1,self.iconv2,self.iconv3,self.iconv4


    def build_resASPPNet_refine(self, model_input):

        conv = self.conv
        if self.params.use_deconv:
            upconv = self.deconv
        else:
            upconv = self.upconv

        with tf.variable_scope('encoder_refine'):
            conv1 = conv(model_input, 64, 7, 2)
            pool1 = self.maxpool(conv1, 3)
            conv2 = self.resblock(pool1, 64, 3)
            conv3 = self.resblock(conv2, 128, 4)
            pool3 = self.maxpool(conv3, 3)
            conv4 = self.ASPPModule(pool3)

        with tf.variable_scope('skips_refine'):
            skip1 = conv1
            skip2 = pool1
            skip3 = conv2
            skip4 = conv3

        # DECODING
        with tf.variable_scope('decoder_refine'):
            upconv5 = upconv(conv4, 256, 3, 2)
            # self.resoftmap5 = self.NonLocalBlockRec(upconv5,16,32, scope='nonlocal_re2')
            concat5 = tf.concat([upconv5, skip4], 3)
            self.reiconv5 = conv(concat5, 256, 3, 1)

            upconv4 = upconv(self.reiconv5, 128, 3, 2)
            # self.resoftmap4 = self.NonLocalBlockRec(upconv4, 16, 32, scope='nonlocal_re3')
            concat4 = tf.concat([upconv4, skip3], 3)
            self.reiconv4 = conv(concat4, 128, 3, 1)
            self.refine_disp4 = self.get_disp(self.reiconv4)
            udisp4 = self.upsample_nn(self.refine_disp4, 2)

            upconv3 = upconv(self.reiconv4, 64, 3, 2)
            # self.resoftmap3 = self.NonLocalBlockRec(upconv3, 16, 32, scope='nonlocal_re4')
            concat3 = tf.concat([upconv3, skip2, udisp4], 3)
            self.reiconv3 = conv(concat3, 64, 3, 1)
            self.refine_disp3 = self.get_disp(self.reiconv3)
            udisp3 = self.upsample_nn(self.refine_disp3, 2)

            upconv2 = upconv(self.reiconv3, 32, 3, 2)
            concat2 = tf.concat([upconv2, skip1, udisp3], 3)
            self.reiconv2 = conv(concat2, 32, 3, 1)
            self.refine_disp2 = self.get_disp(self.reiconv2)
            udisp2 = self.upsample_nn(self.refine_disp2, 2)

            upconv1 = upconv(self.reiconv2, 16, 3, 2)
            concat1 = tf.concat([upconv1, udisp2], 3)
            self.reiconv1 = conv(concat1, 16, 3, 1)
            self.refine_disp1 = self.get_disp(self.reiconv1)
            refine_fea = self.Refindisp(self.refine_disp1, self.reiconv1)

            unfold_disp = self.Refindisp_ex(self.refine_disp1)
            self.refine_disp1 = self.get_disp(refine_fea*unfold_disp)

        return self.refine_disp1, self.refine_disp2, self.refine_disp3, self.refine_disp4, self.reiconv1,self.reiconv2,self.reiconv3,self.reiconv4

    def build_vggNet(self):

        conv = self.conv
        if self.params.use_deconv:
            upconv = self.deconv
        else:
            upconv = self.upconv

        with tf.variable_scope('encoder'):
            conv1 = self.conv_block(self.model_input,  32, 7)
            conv2 = self.conv_block(conv1,             64, 5)
            conv3 = self.conv_block(conv2,            128, 3)
            conv4 = self.conv_block(conv3,            256, 3)
            conv5 = self.conv_block(conv4,            512, 3)
            conv6 = self.conv_block(conv5,            512, 3)
            conv7 = self.conv_block(conv6,            512, 3)

        with tf.variable_scope('skips'):
            skip1 = conv1
            skip2 = conv2
            skip3 = conv3
            skip4 = conv4
            skip5 = conv5
            skip6 = conv6
        
        with tf.variable_scope('decoder'):
            upconv7 = upconv(conv7,  512, 3, 2)
            concat7 = tf.concat([upconv7, skip6], 3)
            iconv7  = conv(concat7,  512, 3, 1)

            upconv6 = upconv(iconv7, 512, 3, 2)
            concat6 = tf.concat([upconv6, skip5], 3)
            iconv6  = conv(concat6,  512, 3, 1)

            upconv5 = upconv(iconv6, 256, 3, 2)
            concat5 = tf.concat([upconv5, skip4], 3)
            iconv5  = conv(concat5,  256, 3, 1)

            upconv4 = upconv(iconv5, 128, 3, 2)
            concat4 = tf.concat([upconv4, skip3], 3)
            iconv4  = conv(concat4,  128, 3, 1)
            self.disp4 = self.get_disp(iconv4)
            udisp4  = self.upsample_nn(self.disp4, 2)

            upconv3 = upconv(iconv4,  64, 3, 2)
            concat3 = tf.concat([upconv3, skip2, udisp4], 3)
            iconv3  = conv(concat3,   64, 3, 1)
            self.disp3 = self.get_disp(iconv3)
            udisp3  = self.upsample_nn(self.disp3, 2)

            upconv2 = upconv(iconv3,  32, 3, 2)
            concat2 = tf.concat([upconv2, skip1, udisp3], 3)
            iconv2  = conv(concat2,   32, 3, 1)
            self.disp2 = self.get_disp(iconv2)
            udisp2  = self.upsample_nn(self.disp2, 2)

            upconv1 = upconv(iconv2,  16, 3, 2)
            concat1 = tf.concat([upconv1, udisp2], 3)
            iconv1  = conv(concat1,   16, 3, 1)
            self.disp1 = self.get_disp(iconv1)


    def build_resNet(self):

        conv   = self.conv
        if self.params.use_deconv:
            upconv = self.deconv
        else:
            upconv = self.upconv

        with tf.variable_scope('encoder'):
            conv1 = conv(self.model_input, 64, 7, 2)
            pool1 = self.maxpool(conv1,           3)
            conv2 = self.resblock(pool1,      64, 3)
            conv3 = self.resblock(conv2,     128, 4)
            conv4 = self.resblock(conv3,     256, 6)
            conv5 = self.resblock(conv4,     512, 3)

        with tf.variable_scope('skips'):
            skip1 = conv1
            skip2 = pool1
            skip3 = conv2
            skip4 = conv3
            skip5 = conv4
        
        # DECODING
        with tf.variable_scope('decoder'):
            upconv6 = upconv(conv5,   512, 3, 2)
            concat6 = tf.concat([upconv6, skip5], 3)
            iconv6  = conv(concat6,   512, 3, 1)

            upconv5 = upconv(iconv6, 256, 3, 2)
            concat5 = tf.concat([upconv5, skip4], 3)
            iconv5  = conv(concat5,   256, 3, 1)

            upconv4 = upconv(iconv5,  128, 3, 2)
            concat4 = tf.concat([upconv4, skip3], 3)
            iconv4  = conv(concat4,   128, 3, 1)
            self.disp4 = self.get_disp(iconv4)
            udisp4  = self.upsample_nn(self.disp4, 2)

            upconv3 = upconv(iconv4,   64, 3, 2)
            concat3 = tf.concat([upconv3, skip2, udisp4], 3)
            iconv3  = conv(concat3,    64, 3, 1)
            self.disp3 = self.get_disp(iconv3)
            udisp3  = self.upsample_nn(self.disp3, 2)

            upconv2 = upconv(iconv3,   32, 3, 2)
            concat2 = tf.concat([upconv2, skip1, udisp3], 3)
            iconv2  = conv(concat2,    32, 3, 1)
            self.disp2 = self.get_disp(iconv2)
            udisp2  = self.upsample_nn(self.disp2, 2)

            upconv1 = upconv(iconv2,  16, 3, 2)
            concat1 = tf.concat([upconv1, udisp2], 3)
            iconv1  = conv(concat1,   16, 3, 1)
            self.disp1 = self.get_disp(iconv1)

    def build_model(self):
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose], activation_fn=tf.nn.elu):
            with tf.variable_scope('model', reuse=tf.AUTO_REUSE):

                self.left_pyramid  = self.scale_pyramid(self.left,  4)
                self.right_pyramid = self.scale_pyramid(self.right, 4)
                self.model_input = self.left

                if self.params.encoder == 'vggNet':
                    self.build_vggNet()
                elif self.params.encoder == 'resNet':
                    self.build_resNet()
                elif self.params.encoder == 'vggASPPNet':
                    self.build_vggASPPNet()
                elif self.params.encoder == 'resASPPNet':
                    with tf.variable_scope('resNet'):
                        self.disp1, self.disp2, self.disp3, self.disp4, \
                        self.iconv1,self.iconv2,self.iconv3,self.iconv4= self.build_resASPPNet()
                        # print('The iteration number')
                        # print(self.params.iter_number)

                    with tf.variable_scope('resNet_refine'):

                        self.output_dict = {}
                        self.output_dict['re_disp1'] = []
                        self.output_dict['re_disp2'] = []
                        self.output_dict['re_disp3'] = []
                        self.output_dict['re_disp4'] = []
                        itera_disp = self.disp1

                        for ii in range(0, self.params.iter_number):

                            model_input = tf.concat([self.left, self.right, itera_disp], 3)
                            self.refine_disp1, self.refine_disp2, self.refine_disp3, self.refine_disp4, \
                            self.reiconv1, self.reiconv2, self.reiconv3, self.reiconv4 = self.build_resASPPNet_refine(
                                model_input)

                            if ii == 0:
                                self.output_dict['re_disp1'].append(self.refine_disp1)
                                self.output_dict['re_disp2'].append(self.refine_disp2)
                                self.output_dict['re_disp3'].append(self.refine_disp3)
                                self.output_dict['re_disp4'].append(self.refine_disp4)

                            else:
                                self.output_dict['re_disp1'].append(self.refine_disp1 + self.output_dict['re_disp1'][ii - 1])
                                self.output_dict['re_disp2'].append(self.refine_disp2 + self.output_dict['re_disp2'][ii - 1])
                                self.output_dict['re_disp3'].append(self.refine_disp3 + self.output_dict['re_disp3'][ii - 1])
                                self.output_dict['re_disp4'].append(self.refine_disp4 + self.output_dict['re_disp4'][ii - 1])

                            if ii < (self.params.iter_number - 1):
                                itera_disp = self.output_dict['re_disp1'][ii]


                else:
                    return None

    def build_outputs(self):

        with tf.variable_scope('disparities'):
            self.disp_est  = [self.disp1, self.disp2, self.disp3, self.disp4]
            self.disp_left_est  = [tf.expand_dims(d[:,:,:,0], 3) for d in self.disp_est]
            self.disp_right_est = [tf.expand_dims(d[:,:,:,1], 3) for d in self.disp_est]
            self.monofeature = [self.iconv1, self.iconv2, self.iconv3, self.iconv4]

        with tf.variable_scope('disparities_refine'):
            self.refine_disp_est = [self.output_dict['re_disp1'][self.params.iter_number-1],self.output_dict['re_disp2'][self.params.iter_number-1],
                                    self.output_dict['re_disp3'][self.params.iter_number - 1],self.output_dict['re_disp4'][self.params.iter_number-1]]
            self.refine_disp_left_est = [tf.expand_dims(d[:, :, :, 0], 3) for d in self.refine_disp_est]
            self.refine_disp_right_est = [tf.expand_dims(d[:, :, :, 1], 3) for d in self.refine_disp_est]
            self.stereofeature = [self.reiconv1, self.reiconv2, self.reiconv3, self.reiconv4]

            self.re_disp_est = []
            for i in range(0, self.params.iter_number):
                re_disp_est_new = [self.output_dict['re_disp1'][i],self.output_dict['re_disp2'][i],
                                           self.output_dict['re_disp3'][i],self.output_dict['re_disp4'][i]]
                self.re_disp_est.append(re_disp_est_new)

            self.re_disp_left_est = []
            self.re_disp_right_est = []
            for i in range(0, self.params.iter_number):
                refine_disp_left_est_tmp = [tf.expand_dims(d[:, :, :, 0], 3) for d in self.re_disp_est[i]]
                self.re_disp_left_est.append(refine_disp_left_est_tmp)

                refine_disp_right_est_tmp = [tf.expand_dims(d[:, :, :, 1], 3) for d in self.re_disp_est[i]]
                self.re_disp_right_est.append(refine_disp_right_est_tmp)


        if self.mode == 'test':

            self.disp_s = self.Final_disp_S(self.output_dict['re_disp1'][self.params.iter_number-1])
            self.disp_m = self.Final_disp_M(self.disp1)
            return

        with tf.variable_scope('images'):
            self.left_est  = [self.generate_image_left(self.right_pyramid[i], self.disp_left_est[i])  for i in range(4)]
            self.right_est = [self.generate_image_right(self.left_pyramid[i], self.disp_right_est[i]) for i in range(4)]

        with tf.variable_scope('images_refine'):

            self.left_est_refine_new=[]
            self.right_est_refine_new=[]
            for j in range(0, self.params.iter_number):
                left_est_refine_new = [self.generate_image_left(self.right_pyramid[i], self.re_disp_left_est[j][i])
                                        for i in range(4)]
                self.left_est_refine_new.append(left_est_refine_new)

                right_est_refine_new = [self.generate_image_right(self.left_pyramid[i], self.re_disp_right_est[j][i])
                                        for i in range(4)]
                self.right_est_refine_new.append(right_est_refine_new)

            self.left_est_refine  = [self.generate_image_left(self.right_pyramid[i], self.refine_disp_left_est[i])  for i in range(4)]
            self.right_est_refine = [self.generate_image_right(self.left_pyramid[i], self.refine_disp_right_est[i]) for i in range(4)]


        with tf.variable_scope('left-right'):
            self.right_to_left_disp = [self.generate_image_left(self.disp_right_est[i], self.disp_left_est[i])  for i in range(4)]
            self.left_to_right_disp = [self.generate_image_right(self.disp_left_est[i], self.disp_right_est[i]) for i in range(4)]

        with tf.variable_scope('left-right_refine'):

            self.right_to_left_disp_refine_new=[]
            self.left_to_right_disp_refine_new=[]
            for j in range(0, self.params.iter_number):
                right_to_left_disp_refine_new = [self.generate_image_left(self.re_disp_right_est[j][i], self.re_disp_left_est[j][i])
                                        for i in range(4)]
                self.right_to_left_disp_refine_new.append(right_to_left_disp_refine_new)

                left_to_right_disp_refine_new = [self.generate_image_right(self.re_disp_left_est[j][i], self.re_disp_right_est[j][i])
                                        for i in range(4)]
                self.left_to_right_disp_refine_new.append(left_to_right_disp_refine_new)

            self.right_to_left_disp_refine = [self.generate_image_left(self.refine_disp_right_est[i], self.refine_disp_left_est[i])  for i in range(4)]
            self.left_to_right_disp_refine = [self.generate_image_right(self.refine_disp_left_est[i], self.refine_disp_right_est[i]) for i in range(4)]


        with tf.variable_scope('smoothness'):
            self.disp_left_smoothness  = self.get_smoothness(self.disp_left_est,  self.left_pyramid)
            self.disp_right_smoothness = self.get_smoothness(self.disp_right_est, self.right_pyramid)

        with tf.variable_scope('smoothness_refine'):
            self.disp_left_smoothness_refine = self.get_smoothness(self.refine_disp_left_est, self.left_pyramid)
            self.disp_right_smoothness_refine = self.get_smoothness(self.refine_disp_right_est, self.right_pyramid)

    def build_losses(self):
        with tf.variable_scope('losses', reuse=tf.AUTO_REUSE):
            # # Long-Range (Non-Local) Dependencies Distillation
            # self.softm = [self.softmap5,self.softmap4]
            # self.resoftm = [self.resoftmap5,self.resoftmap4]
            # self.simi_dis_loss_left_temp = [
            #     tf.reduce_mean(tf.abs(self.softm[i] - self.resoftm[i])) for i in
            #     range(2)]
            # self.simi_dis_loss_left = tf.add_n(self.simi_dis_loss_left_temp)

            # Output Space Distillation
            self.errormap = tf.abs(self.disp_left_est[0] - tf.stop_gradient(self.refine_disp_left_est[0]))
            self.distill_loss_left = [
                tf.reduce_mean(tf.abs(self.disp_left_est[i] - tf.stop_gradient(self.refine_disp_left_est[i]))) for i in
                range(4)]
            # self.errormap = tf.abs(self.disp_left_est[0] - self.refine_disp_left_est[0])
            # self.distill_loss_left = [
            #     tf.reduce_mean(tf.abs(self.disp_left_est[i] - self.refine_disp_left_est[i])) for i in
            #     range(4)]
            self.distill_loss_left = tf.add_n(self.distill_loss_left)
            ##Feature Space Distillation
            self.perceptual_loss_left = [
                tf.reduce_mean(tf.abs(self.monofeature[i] - self.stereofeature[i])) for i in
                range(4)]
            self.perceptual_loss_left = tf.add_n(self.perceptual_loss_left)
            # IMAGE RECONSTRUCTION
            self.l1_left = [tf.abs( self.left_est[i] - self.left_pyramid[i]) for i in range(4)]
            self.l1_reconstruction_loss_left  = [tf.reduce_mean(l) for l in self.l1_left]
            self.l1_right = [tf.abs(self.right_est[i] - self.right_pyramid[i]) for i in range(4)]
            self.l1_reconstruction_loss_right = [tf.reduce_mean(l) for l in self.l1_right]

            self.l1_reconstruction_loss_left_refine_new=[]
            self.l1_reconstruction_loss_right_refine_new=[]
            for j in range(0, self.params.iter_number):
                l1_left_refine_new = [tf.abs(self.left_est_refine_new[j][i] - self.left_pyramid[i]) for i in range(4)]
                l1_reconstruction_loss_left_refine_new = [tf.reduce_mean(l) for l in l1_left_refine_new]
                self.l1_reconstruction_loss_left_refine_new.append(l1_reconstruction_loss_left_refine_new)

                l1_right_refine_new = [tf.abs(self.right_est_refine_new[j][i] - self.right_pyramid[i]) for i in range(4)]
                l1_reconstruction_loss_right_refine_new = [tf.reduce_mean(l) for l in l1_right_refine_new]
                self.l1_reconstruction_loss_right_refine_new.append(l1_reconstruction_loss_right_refine_new)

            self.l1_rec_loss_left_refine = []
            self.l1_rec_loss_right_refine = []
            for i in range(0, 4):
                l1_rec_loss_left_refine_new = [self.l1_reconstruction_loss_left_refine_new[j][i] for j in range(self.params.iter_number)]
                self.l1_rec_loss_left_refine.append(tf.reduce_mean(l1_rec_loss_left_refine_new))

                l1_rec_loss_right_refine_new = [self.l1_reconstruction_loss_right_refine_new[j][i] for j in range(self.params.iter_number)]
                self.l1_rec_loss_right_refine.append(tf.reduce_mean(l1_rec_loss_right_refine_new))



            self.ssim_left = [self.SSIM( self.left_est[i],  self.left_pyramid[i]) for i in range(4)]
            self.ssim_loss_left  = [tf.reduce_mean(s) for s in self.ssim_left]
            self.ssim_right = [self.SSIM(self.right_est[i], self.right_pyramid[i]) for i in range(4)]
            self.ssim_loss_right = [tf.reduce_mean(s) for s in self.ssim_right]

            self.ssim_left_refine_new=[]
            self.ssim_right_refine_new=[]
            for j in range(0, self.params.iter_number):
                ssim_left_refine_new = [self.SSIM(self.left_est_refine_new[j][i], self.left_pyramid[i]) for i in range(4)]
                ssim_loss_left_refine_new = [tf.reduce_mean(s) for s in ssim_left_refine_new]
                self.ssim_left_refine_new.append(ssim_loss_left_refine_new)

                ssim_right_refine_new = [self.SSIM(self.right_est_refine_new[j][i], self.right_pyramid[i]) for i in range(4)]
                ssim_loss_right_refine_new = [tf.reduce_mean(s) for s in ssim_right_refine_new]
                self.ssim_right_refine_new.append(ssim_loss_right_refine_new)

            self.ossim_left_refine_new = []
            self.ossim_right_refine_new = []
            for i in range(0, 4):
                ossim_left_refine_new = [self.ssim_left_refine_new[j][i] for j in range(self.params.iter_number)]
                self.ossim_left_refine_new.append(tf.reduce_mean(ossim_left_refine_new))

                ossim_right_refine_new = [self.ssim_right_refine_new[j][i] for j in range(self.params.iter_number)]
                self.ossim_right_refine_new.append(tf.reduce_mean(ossim_right_refine_new))


            self.image_loss_right = [self.params.alpha_image_loss * self.ssim_loss_right[i] + (1 - self.params.alpha_image_loss) * self.l1_reconstruction_loss_right[i] for i in range(4)]
            self.image_loss_left  = [self.params.alpha_image_loss * self.ssim_loss_left[i]  + (1 - self.params.alpha_image_loss) * self.l1_reconstruction_loss_left[i]  for i in range(4)]
            self.image_loss = tf.add_n(self.image_loss_left + self.image_loss_right)

            self.image_loss_right_refine = [self.params.alpha_image_loss * self.ossim_right_refine_new[i] + (1 - self.params.alpha_image_loss) * self.l1_rec_loss_right_refine[i] for i in range(4)]
            self.image_loss_left_refine  = [self.params.alpha_image_loss * self.ossim_left_refine_new[i]  + (1 - self.params.alpha_image_loss) * self.l1_rec_loss_left_refine[i]  for i in range(4)]
            self.image_loss_refine = tf.add_n(self.image_loss_left_refine + self.image_loss_right_refine)

            # DISPARITY SMOOTHNESS
            self.disp_left_loss  = [tf.reduce_mean(tf.abs(self.disp_left_smoothness[i]))  / 2 ** i for i in range(4)]
            self.disp_right_loss = [tf.reduce_mean(tf.abs(self.disp_right_smoothness[i])) / 2 ** i for i in range(4)]
            self.disp_gradient_loss = tf.add_n(self.disp_left_loss + self.disp_right_loss)

            self.disp_left_loss_refine = [tf.reduce_mean(tf.abs(self.disp_left_smoothness_refine[i])) / 2 ** i for i in range(4)]
            self.disp_right_loss_refine = [tf.reduce_mean(tf.abs(self.disp_right_smoothness_refine[i])) / 2 ** i for i in range(4)]
            self.disp_gradient_loss_refine = tf.add_n(self.disp_left_loss_refine + self.disp_right_loss_refine)

            # LR CONSISTENCY
            self.lr_left_loss  = [tf.reduce_mean(tf.abs(self.right_to_left_disp[i] - self.disp_left_est[i]))  for i in range(4)]
            self.lr_right_loss = [tf.reduce_mean(tf.abs(self.left_to_right_disp[i] - self.disp_right_est[i])) for i in range(4)]
            self.lr_loss = tf.add_n(self.lr_left_loss + self.lr_right_loss)


            self.lr_left_loss_refine_new=[]
            self.lr_right_loss_refine_new=[]
            for j in range(0, self.params.iter_number):
                lr_left_loss_refine_new = [tf.abs(self.right_to_left_disp_refine_new[j][i] - self.re_disp_left_est[j][i]) for i in range(4)]
                add_lr_left_loss_refine_new = [tf.reduce_mean(l) for l in lr_left_loss_refine_new]
                self.lr_left_loss_refine_new.append(tf.add_n(add_lr_left_loss_refine_new))

                lr_right_loss_refine_new = [tf.abs(self.left_to_right_disp_refine_new[j][i] - self.re_disp_right_est[j][i]) for i in range(4)]
                add_lr_right_loss_refine_new = [tf.reduce_mean(l) for l in lr_right_loss_refine_new]
                self.lr_right_loss_refine_new.append(tf.add_n(add_lr_right_loss_refine_new))

            self.lr_loss_refine = tf.add_n(self.lr_left_loss_refine_new + self.lr_right_loss_refine_new)/self.params.iter_number




            ## TOTAL LOSS
            # Mono-Net Loss(1)
            # self.total_loss = self.image_loss + self.params.disp_gradient_loss_weight * self.disp_gradient_loss + self.params.lr_loss_weight * self.lr_loss
            # Mono-Net Loss + Stereo-Net Loss(2)
            self.total_loss = self.image_loss + self.params.disp_gradient_loss_weight * self.disp_gradient_loss + self.params.lr_loss_weight * self.lr_loss + self.image_loss_refine + self.params.disp_gradient_loss_weight * self.disp_gradient_loss_refine + self.params.lr_loss_weight * self.lr_loss_refine
            # Total Loss + Distillation Loss(3)
            # self.total_loss =  self.image_loss*0.7 + self.params.disp_gradient_loss_weight * self.disp_gradient_loss + self.params.lr_loss_weight * self.lr_loss + self.image_loss_refine*0.8 + self.params.disp_gradient_loss_weight * self.disp_gradient_loss_refine + self.params.lr_loss_weight * self.lr_loss_refine+ self.distill_loss_left * 1 +self.simi_dis_loss_left*1#+ self.perceptual_loss_left*0.1



