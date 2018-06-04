# -*- coding: utf-8 -*-
"""
Created on Sun Aug  6 21:36:50 2017

@author: XLP
"""
import tensorflow as tf
import tensorlayer as tl
import numpy as np
from tensorlayer.layers import list_remove_repeat
Layer = tl.layers.Layer

def image_preprocess(img,meanval):
    meanval = tf.constant(meanval,tf.float32)* (1./255)
    img = tf.cast(img, tf.float32) 
    img = tf.subtract(img, meanval)
    return img

class Mergelayer(Layer):
    def __init__(
        self,
        layer = [],
        name ='merge_layer',
    ):
        Layer.__init__(self, name=name)
        
        self.all_layers = list(layer[0].all_layers)
        self.all_params = list(layer[0].all_params)
        self.all_drop = dict(layer[0].all_drop)

        for i in range(1, len(layer)):
            self.all_layers.extend(list(layer[i].all_layers))
            self.all_params.extend(list(layer[i].all_params))
            self.all_drop.update(dict(layer[i].all_drop))

        self.all_layers = list_remove_repeat(self.all_layers)
        self.all_params = list_remove_repeat(self.all_params)    
                                 
   
                             
def model_VGG16_DIMAN(x, y_,fw, im, iw, mm, mw, c, batch_size, data_shape, reuse,mean_file_name=None,is_train = True, network_scopename = "VGG16_DIMAVN" ):
    if mean_file_name!=None:
        meanval = np.load(mean_file_name)
        x = image_preprocess(x,meanval)
        
    gamma_init=tf.random_normal_initializer(2., 0.1)
    with tf.variable_scope(network_scopename, reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        network_input = tl.layers.InputLayer(x, name='input')
        """ conv1 """
        network1 = tl.layers.Conv2dLayer(network_input, shape = [3, 3, 3, 64],
                    W_init=tf.contrib.layers.xavier_initializer_conv2d(uniform=True, seed=None, dtype=tf.float32),
                    b_init = tf.constant_initializer(value=0.0),
                    strides = [1, 1, 1, 1], padding='SAME', name ='conv1_1')
        network1 = tl.layers.Conv2dLayer(network1, shape = [3, 3, 64, 64],    # 64 features for each 3x3 patch
                    W_init=tf.contrib.layers.xavier_initializer_conv2d(uniform=True, seed=None, dtype=tf.float32),
                    b_init = tf.constant_initializer(value=0.0),
                    strides = [1, 1, 1, 1], padding='SAME', name ='conv1_2')
        network1 = tl.layers.BatchNormLayer(network1, act = tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='bn1')
        network1 = tl.layers.PoolLayer(network1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],padding='SAME', 
                    pool = tf.nn.max_pool, name ='pool1') #outputsize: [H/2,W/2]
        """ conv2 """
        network2 = tl.layers.Conv2dLayer(network1, shape = [3, 3, 64, 128],  # 128 features for each 3x3 patch
                    W_init=tf.contrib.layers.xavier_initializer_conv2d(uniform=True, seed=None, dtype=tf.float32),
                    b_init = tf.constant_initializer(value=0.0),
                                         strides = [1, 1, 1, 1], padding='SAME', name ='conv2_1')
        network2 = tl.layers.Conv2dLayer(network2, shape = [3, 3, 128, 128],  # 128 features for each 3x3 patch
                    W_init=tf.contrib.layers.xavier_initializer_conv2d(uniform=True, seed=None, dtype=tf.float32),
                    b_init = tf.constant_initializer(value=0.0),
                                         strides = [1, 1, 1, 1], padding='SAME', name ='conv2_2')
        network2 = tl.layers.BatchNormLayer(network2, act = tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='bn2')
        network2 = tl.layers.PoolLayer(network2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                    padding='SAME', pool = tf.nn.max_pool, name ='pool2') #outputsize: [H/4,W/4]
        """ conv3 """
        network3 = tl.layers.Conv2dLayer(network2, shape = [3, 3, 128, 256],  # 256 features for each 3x3 patch
                    W_init=tf.contrib.layers.xavier_initializer_conv2d(uniform=True, seed=None, dtype=tf.float32),
                    b_init = tf.constant_initializer(value=0.0),
                                         strides = [1, 1, 1, 1], padding='SAME', name ='conv3_1')
        network3 = tl.layers.Conv2dLayer(network3, shape = [3, 3, 256, 256],  # 256 features for each 3x3 patch
                    W_init=tf.contrib.layers.xavier_initializer_conv2d(uniform=True, seed=None, dtype=tf.float32),
                    b_init = tf.constant_initializer(value=0.0),
                                         strides = [1, 1, 1, 1], padding='SAME', name ='conv3_2')
        network3 = tl.layers.Conv2dLayer(network3, shape = [3, 3, 256, 256],  # 256 features for each 3x3 patch
                    W_init=tf.contrib.layers.xavier_initializer_conv2d(uniform=True, seed=None, dtype=tf.float32),
                    b_init = tf.constant_initializer(value=0.0),
                                         strides = [1, 1, 1, 1], padding='SAME', name ='conv3_3')
        network3 = tl.layers.BatchNormLayer(network3, act = tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='bn3')
        network3 = tl.layers.PoolLayer(network3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                    padding='SAME', pool = tf.nn.max_pool, name ='pool3') #outputsize: [H/8,W/8]
        """ conv4 """
        network4 = tl.layers.Conv2dLayer(network3, shape = [3, 3, 256, 512],  # 512 features for each 3x3 patch
                    W_init=tf.contrib.layers.xavier_initializer_conv2d(uniform=True, seed=None, dtype=tf.float32),
                    b_init = tf.constant_initializer(value=0.0),
                                         strides = [1, 1, 1, 1], padding='SAME', name ='conv4_1')
        network4 = tl.layers.Conv2dLayer(network4, shape = [3, 3, 512, 512],  # 512 features for each 3x3 patch
                    W_init=tf.contrib.layers.xavier_initializer_conv2d(uniform=True, seed=None, dtype=tf.float32),
                    b_init = tf.constant_initializer(value=0.0),
                                         strides = [1, 1, 1, 1], padding='SAME', name ='conv4_2')
        network4 = tl.layers.Conv2dLayer(network4, shape = [3, 3, 512, 512],  # 512 features for each 3x3 patch
                    strides = [1, 1, 1, 1], padding='SAME', name ='conv4_3')
        network4 = tl.layers.BatchNormLayer(network4, act = tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='bn4')
        network4 = tl.layers.PoolLayer(network4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                    padding='SAME', pool = tf.nn.max_pool, name ='pool4') #outputsize: [H/16,W/16]
        """ conv5 """
        network5 = tl.layers.Conv2dLayer(network4, shape = [3, 3, 512, 512],  # 512 features for each 3x3 patch
                    W_init=tf.contrib.layers.xavier_initializer_conv2d(uniform=True, seed=None, dtype=tf.float32),
                    b_init = tf.constant_initializer(value=0.0),
                                         strides = [1, 1, 1, 1], padding='SAME', name ='conv5_1')
        network5 = tl.layers.Conv2dLayer(network5, shape = [3, 3, 512, 512],  # 512 features for each 3x3 patch
                    W_init=tf.contrib.layers.xavier_initializer_conv2d(uniform=True, seed=None, dtype=tf.float32),
                    b_init = tf.constant_initializer(value=0.0),
                                         strides = [1, 1, 1, 1], padding='SAME', name ='conv5_2')
        network5 = tl.layers.Conv2dLayer(network5, shape = [3, 3, 512, 512],  # 512 features for each 3x3 patch
                    W_init=tf.contrib.layers.xavier_initializer_conv2d(uniform=True, seed=None, dtype=tf.float32),
                    b_init = tf.constant_initializer(value=0.0),
                                         strides = [1, 1, 1, 1], padding='SAME', name ='conv5_3')
        network5 = tl.layers.BatchNormLayer(network5, act = tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='bn5')
        network5 = tl.layers.PoolLayer(network5, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                    padding='SAME', pool = tf.nn.max_pool, name ='pool5') #outputsize: [H/32,W/32]
        
        '#########################Upsample and merge##########################'
        '''top-down 5'''
        network5_conv = tl.layers.Conv2dLayer(network5, shape = [3, 3, 512, 64], act = tf.nn.relu,  # 512 features for each 3x3 patch
                    strides = [1, 1, 1, 1], padding='SAME', name ='conv5_conv')
        network5_up = tl.layers.UpSampling2dLayer(network5_conv,
                    size = [data_shape[0]//16,data_shape[1]//16],method =0,is_scale = False,name = 'upsample5' )   # output:[H/16,W/16,64]
        '''top-down 4'''
        network4_conv = tl.layers.Conv2dLayer(network4, shape = [3, 3, 512, 64], act = tf.nn.relu, # 512 features for each 3x3 patch
                    strides = [1, 1, 1, 1], padding='SAME', name ='conv4_conv')
        network_cmb4_5 = tl.layers.ConcatLayer([network4_conv,network5_up],
                    concat_dim = 3, name = 'concat_4_5') # output:[H/16,W/16,128]
        network4_up = tl.layers.UpSampling2dLayer(network_cmb4_5, 
                    size = [data_shape[0]//8,data_shape[1]//8], method =0,is_scale = False,name = 'upsample4' )   # output:[H/8,W/8,128]
        '''top-down 3'''
        network3_conv = tl.layers.Conv2dLayer(network3, shape = [3, 3, 256, 64], act = tf.nn.relu, # 512 features for each 3x3 patch
                    strides = [1, 1, 1, 1], padding='SAME', name ='conv3_conv') # output:[H/8,W/8,64]
        network_cmb3_4 = tl.layers.ConcatLayer([network3_conv,network4_up],
                    concat_dim = 3, name = 'concat_3_4')# output:[H/8,W/8,192]
        network3_up = tl.layers.UpSampling2dLayer(network_cmb3_4, 
                    size = [data_shape[0]//4,data_shape[1]//4], method =0,is_scale = False,name = 'upsample3' )   # output:[H/4,W/4,192]
        '''top-down 2'''
        network2_conv = tl.layers.Conv2dLayer(network2, shape = [3, 3, 128, 64],act = tf.nn.relu,  # 512 features for each 3x3 patch
                    strides = [1, 1, 1, 1], padding='SAME', name ='conv2_conv') # output:[H/4,W/4,64]
        network_cmb2_3 = tl.layers.ConcatLayer([network2_conv,network3_up],
                    concat_dim = 3, name = 'concat_2_3')# output:[H/4,W/4,256]
        network2_up = tl.layers.UpSampling2dLayer(network_cmb2_3, 
                    size = [data_shape[0]//2,data_shape[1]//2], method =0,is_scale = False,name = 'upsample2' )   # output:[H/2,W/2,256]

        '''top-down 1'''
        network1_conv = tl.layers.Conv2dLayer(network1, shape = [3, 3, 64, 64], act = tf.nn.relu, # 512 features for each 3x3 patch
                    strides = [1, 1, 1, 1], padding='SAME', name ='conv1_conv') # output:[H/2,W/2,64]
        network_cmb1_2 = tl.layers.ConcatLayer([network1_conv,network2_up],
                    concat_dim = 3, name = 'concat1_2')# output:[H/2,W/2,320]
        network1_up = tl.layers.UpSampling2dLayer(network_cmb1_2, 
                    size = [data_shape[0],data_shape[1]], method =0,is_scale = False,name = 'upsample1' )   # output:[H,W,320]
        
                                                                           
        
        """## cost of classification3"""
        network_foreground = tl.layers.Conv2dLayer(network1_up,             
               shape = [3, 3, 320, 160], # 64 features for each 5x5 patch
               strides=[1, 1, 1, 1],
               padding='SAME',
               W_init=tf.contrib.layers.xavier_initializer_conv2d(uniform=True, seed=None, dtype=tf.float32),
               b_init = tf.constant_initializer(value=0.0),
               name ='score3_feaconv')  # output: (?, 14, 14, 64)
        network_foreground = tl.layers.Conv2dLayer(network_foreground,             
               shape = [3, 3, 160, 2], # 64 features for each 5x5 patch
               strides=[1, 1, 1, 1],
               padding='SAME',
               W_init=tf.contrib.layers.xavier_initializer_conv2d(uniform=True, seed=None, dtype=tf.float32),
               b_init = tf.constant_initializer(value=0.0),
               name ='output3')  # output: (?, 14, 14, 64)
        
        network_interval = tl.layers.Conv2dLayer(network2_up,             
               shape = [3, 3, 256, 128], # 64 features for each 5x5 patch
               strides=[1, 1, 1, 1],
               padding='SAME',
               W_init=tf.contrib.layers.xavier_initializer_conv2d(uniform=True, seed=None, dtype=tf.float32),
               b_init = tf.constant_initializer(value=0.0),
               name ='score_interval_feaconv')  # output: (?, 14, 14, 64)
        network_interval = tl.layers.Conv2dLayer(network_interval,             
               shape = [3, 3, 128, 2], # 64 features for each 5x5 patch
               strides=[1, 1, 1, 1],
               padding='SAME',
               W_init=tf.contrib.layers.xavier_initializer_conv2d(uniform=True, seed=None, dtype=tf.float32),
               b_init = tf.constant_initializer(value=0.0),
               name ='output_interval')  # output: (?, 14, 14, 64)
        network_interval =tl.layers.UpSampling2dLayer(network_interval,
                   size = [data_shape[0],data_shape[1]],
                   method =0,
                   is_scale = False,
                   name = 'output_interval_up' )
        
        network_masker = tl.layers.Conv2dLayer(network3_up,             
               shape = [3, 3, 192, 96], # 64 features for each 5x5 patch
               strides=[1, 1, 1, 1],
               padding='SAME',
               W_init=tf.contrib.layers.xavier_initializer_conv2d(uniform=True, seed=None, dtype=tf.float32),
               b_init = tf.constant_initializer(value=0.0),
               name ='score_masker_feaconv')  # output: (?, 14, 14, 64)
        network_masker = tl.layers.Conv2dLayer(network_masker,             
               shape = [3, 3, 96, 2], # 64 features for each 5x5 patch
               strides=[1, 1, 1, 1],
               padding='SAME',
               W_init=tf.contrib.layers.xavier_initializer_conv2d(uniform=True, seed=None, dtype=tf.float32),
               b_init = tf.constant_initializer(value=0.0),
               name ='output_masker')  # output: (?, 14, 14, 64)
        network_masker =tl.layers.UpSampling2dLayer(network_masker,
                   size = [data_shape[0],data_shape[1]],
                   method =0,
                   is_scale = False,
                   name = 'output_masker_up' )
        if is_train:
            """## cost of classification2"""
    
            
            ## merge all classification
            network_final = Mergelayer([network_foreground,network_interval,network_masker],
                   name = 'mergeall'               )
              
            #================Groundtruth==========================
            y_ = tf.reshape(y_,[batch_size*data_shape[0]*data_shape[1],2])
            fw = tf.reshape(fw,[batch_size*data_shape[0]*data_shape[1]])
            fw = fw*data_shape[0]*data_shape[1]*batch_size/tf.reduce_sum(fw)
			
            im = tf.reshape(im,[batch_size*data_shape[0]*data_shape[1],2])
            iw = tf.reshape(iw,[batch_size*data_shape[0]*data_shape[1]])
            iw = iw*data_shape[0]*data_shape[1]*batch_size/tf.reduce_sum(iw)
            
            mm= tf.reshape(mm,[batch_size*data_shape[0]*data_shape[1],2])
            mw = tf.reshape(mw,[batch_size*data_shape[0]*data_shape[1]])
            mw = mw*data_shape[0]*data_shape[1]*batch_size/tf.reduce_sum(mw)
                        
              
			#================cost foreground================================
            y_foreground = network_foreground.outputs   
            y_foreground = tf.reshape(y_foreground,[batch_size*data_shape[0]*data_shape[1],2])
            y_foreground_prob = tf.nn.softmax(y_foreground,name='softmax_foreground')
            y_foreground_class = tf.argmax(y_foreground_prob, 1)
            y_foreground_class = tf.reshape(y_foreground_class,[batch_size,data_shape[0],data_shape[1],1])
            
            cost_foreground = -tf.reduce_mean(tf.multiply(tf.reduce_sum(y_*tf.log(y_foreground_prob),1),fw))
            
            #================cost interval================================
            y_interval = network_interval.outputs   
            y_interval = tf.reshape(y_interval,[batch_size*data_shape[0]*data_shape[1],2])
            y_interval_prob = tf.nn.softmax(y_interval,name='softmax_interval')
            y_interval_class = tf.argmax(y_interval_prob, 1)
            y_interval_class = tf.reshape(y_interval_class,[batch_size,data_shape[0],data_shape[1],1])
            
            cost_interval = -tf.reduce_mean(tf.multiply(tf.reduce_sum(im*tf.log(y_interval_prob),1),iw))
            
            #================cost interval================================
            y_masker = network_masker.outputs   
            y_masker = tf.reshape(y_masker,[batch_size*data_shape[0]*data_shape[1],2])
            y_masker_prob = tf.nn.softmax(y_masker,name='softmax_interval')
            y_masker_class = tf.argmax(y_masker_prob, 1)          
            y_masker_class = tf.reshape(y_masker_class,[batch_size,data_shape[0],data_shape[1],1])
            
            cost_masker = -tf.reduce_mean(tf.multiply(tf.reduce_sum(mm*tf.log(y_masker_prob),1),mw))
            
            #================fg & interval================================
            y_refine_class = tf.logical_and(tf.cast(y_foreground_class,tf.bool),tf.logical_not(tf.cast(y_interval_class,tf.bool)))
            
            
            
            
            cost = c[0]*cost_foreground+c[1]*cost_interval+c[2]*cost_masker
            #================costall================================
            return network_final,cost,y_foreground_class,y_interval_class, y_masker_class, y_refine_class
        else:
            network_final = Mergelayer([network_foreground,network_interval,network_masker],
                   name = 'mergeall'               )
             
            y_foreground = network_foreground.outputs   
            y_foreground = tf.reshape(y_foreground,[batch_size*data_shape[0]*data_shape[1],2])
            y_foreground_prob = tf.nn.softmax(y_foreground,name='softmax_foreground')
            y_foreground_class = tf.argmax(y_foreground_prob, 1)
            y_foreground_prob = tf.reshape(y_foreground_prob,[batch_size, data_shape[0],data_shape[1],2])
            y_foreground_prob = tf.slice(y_foreground_prob,[0,0,0,1],[batch_size, data_shape[0],data_shape[1],1])
            y_foreground_class = tf.reshape(y_foreground_class,[batch_size, data_shape[0],data_shape[1],1])
            
            y_interval = network_interval.outputs   
            y_interval = tf.reshape(y_interval,[batch_size*data_shape[0]*data_shape[1],2])
            y_interval_prob = tf.nn.softmax(y_interval,name='softmax_interval')
            y_interval_class = tf.argmax(y_interval_prob, 1)
            y_interval_class = tf.reshape(y_interval_class,[batch_size,data_shape[0],data_shape[1],1])
            
            y_masker = network_masker.outputs   
            y_masker = tf.reshape(y_masker,[batch_size*data_shape[0]*data_shape[1],2])
            y_masker_prob = tf.nn.softmax(y_masker,name='softmax_interval')
            y_masker_class = tf.argmax(y_masker_prob, 1)
            y_masker_class = tf.reshape(y_masker_class,[batch_size,data_shape[0],data_shape[1],1])
            
            y_refine_class = tf.logical_and(tf.cast(y_foreground_class,tf.bool),tf.logical_not(tf.cast(y_interval_class,tf.bool)))
            y_refine_prob = tf.multiply(y_foreground_prob,tf.cast(y_refine_class,tf.float32))
            
            return network_final, y_foreground_class, y_foreground_prob, y_interval_class, y_masker_class, y_refine_class, y_refine_prob                               

            
def model_VGG16_FCN8s(x, y_,fw, batch_size, data_shape, reuse=False, mean_file_name=None,is_train = True, network_scopename = "VGG16_FCN8s" ):
    
    if mean_file_name!=None:
        meanval = np.load(mean_file_name)
        x = image_preprocess(x,meanval)
    
    nx = data_shape[0]
    ny = data_shape[1]
  
    gamma_init=tf.random_normal_initializer(2., 0.1)
    with tf.variable_scope(network_scopename, reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        network_input = tl.layers.InputLayer(x, name='input')
        """ conv1 """
        conv1 = tl.layers.Conv2dLayer(network_input, shape = [3, 3, 3, 64],
                    W_init=tf.contrib.layers.xavier_initializer_conv2d(uniform=True, seed=None, dtype=tf.float32),
                    b_init = tf.constant_initializer(value=0.0),
                    strides = [1, 1, 1, 1], padding='SAME', name ='conv1_1')
        conv1 = tl.layers.Conv2dLayer(conv1, shape = [3, 3, 64, 64],    # 64 features for each 3x3 patch
                    W_init=tf.contrib.layers.xavier_initializer_conv2d(uniform=True, seed=None, dtype=tf.float32),
                    b_init = tf.constant_initializer(value=0.0),
                    strides = [1, 1, 1, 1], padding='SAME', name ='conv1_2')
        conv1 = tl.layers.BatchNormLayer(conv1, act = tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='bn1')
        pool1 = tl.layers.PoolLayer(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],padding='SAME', 
                    pool = tf.nn.max_pool, name ='pool1') #outputsize: [H/2,W/2]
        """ conv2 """
        conv2 = tl.layers.Conv2dLayer(pool1, shape = [3, 3, 64, 128],  # 128 features for each 3x3 patch
                    W_init=tf.contrib.layers.xavier_initializer_conv2d(uniform=True, seed=None, dtype=tf.float32),
                    b_init = tf.constant_initializer(value=0.0),
                                         strides = [1, 1, 1, 1], padding='SAME', name ='conv2_1')
        conv2 = tl.layers.Conv2dLayer(conv2, shape = [3, 3, 128, 128],  # 128 features for each 3x3 patch
                    W_init=tf.contrib.layers.xavier_initializer_conv2d(uniform=True, seed=None, dtype=tf.float32),
                    b_init = tf.constant_initializer(value=0.0),
                                         strides = [1, 1, 1, 1], padding='SAME', name ='conv2_2')
        conv2 = tl.layers.BatchNormLayer(conv2, act = tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='bn2')
        pool2 = tl.layers.PoolLayer(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                    padding='SAME', pool = tf.nn.max_pool, name ='pool2') #outputsize: [H/4,W/4]
        """ conv3 """
        conv3 = tl.layers.Conv2dLayer(pool2, shape = [3, 3, 128, 256],  # 256 features for each 3x3 patch
                    W_init=tf.contrib.layers.xavier_initializer_conv2d(uniform=True, seed=None, dtype=tf.float32),
                    b_init = tf.constant_initializer(value=0.0),
                                         strides = [1, 1, 1, 1], padding='SAME', name ='conv3_1')
        conv3 = tl.layers.Conv2dLayer(conv3, shape = [3, 3, 256, 256],  # 256 features for each 3x3 patch
                    W_init=tf.contrib.layers.xavier_initializer_conv2d(uniform=True, seed=None, dtype=tf.float32),
                    b_init = tf.constant_initializer(value=0.0),
                                         strides = [1, 1, 1, 1], padding='SAME', name ='conv3_2')
        conv3 = tl.layers.Conv2dLayer(conv3, shape = [3, 3, 256, 256],  # 256 features for each 3x3 patch
                    W_init=tf.contrib.layers.xavier_initializer_conv2d(uniform=True, seed=None, dtype=tf.float32),
                    b_init = tf.constant_initializer(value=0.0),
                                         strides = [1, 1, 1, 1], padding='SAME', name ='conv3_3')
        conv3 = tl.layers.BatchNormLayer(conv3, act = tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='bn3')
        pool3 = tl.layers.PoolLayer(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                    padding='SAME', pool = tf.nn.max_pool, name ='pool3') #outputsize: [H/8,W/8]
        """ conv4 """
        conv4 = tl.layers.Conv2dLayer(pool3, shape = [3, 3, 256, 512],  # 512 features for each 3x3 patch
                    W_init=tf.contrib.layers.xavier_initializer_conv2d(uniform=True, seed=None, dtype=tf.float32),
                    b_init = tf.constant_initializer(value=0.0),
                                         strides = [1, 1, 1, 1], padding='SAME', name ='conv4_1')
        conv4 = tl.layers.Conv2dLayer(conv4, shape = [3, 3, 512, 512],  # 512 features for each 3x3 patch
                    W_init=tf.contrib.layers.xavier_initializer_conv2d(uniform=True, seed=None, dtype=tf.float32),
                    b_init = tf.constant_initializer(value=0.0),
                                         strides = [1, 1, 1, 1], padding='SAME', name ='conv4_2')
        conv4 = tl.layers.Conv2dLayer(conv4, shape = [3, 3, 512, 512],  # 512 features for each 3x3 patch
                    strides = [1, 1, 1, 1], padding='SAME', name ='conv4_3')
        conv4 = tl.layers.BatchNormLayer(conv4, act = tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='bn4')
        pool4 = tl.layers.PoolLayer(conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                    padding='SAME', pool = tf.nn.max_pool, name ='pool4') #outputsize: [H/16,W/16]
        """ conv5 """
        conv5 = tl.layers.Conv2dLayer(pool4, shape = [3, 3, 512, 512],  # 512 features for each 3x3 patch
                    W_init=tf.contrib.layers.xavier_initializer_conv2d(uniform=True, seed=None, dtype=tf.float32),
                    b_init = tf.constant_initializer(value=0.0),
                                         strides = [1, 1, 1, 1], padding='SAME', name ='conv5_1')
        conv5 = tl.layers.Conv2dLayer(conv5, shape = [3, 3, 512, 512],  # 512 features for each 3x3 patch
                    W_init=tf.contrib.layers.xavier_initializer_conv2d(uniform=True, seed=None, dtype=tf.float32),
                    b_init = tf.constant_initializer(value=0.0),
                                         strides = [1, 1, 1, 1], padding='SAME', name ='conv5_2')
        conv5 = tl.layers.Conv2dLayer(conv5, shape = [3, 3, 512, 512],  # 512 features for each 3x3 patch
                    W_init=tf.contrib.layers.xavier_initializer_conv2d(uniform=True, seed=None, dtype=tf.float32),
                    b_init = tf.constant_initializer(value=0.0),
                                         strides = [1, 1, 1, 1], padding='SAME', name ='conv5_3')
        conv5 = tl.layers.BatchNormLayer(conv5, act = tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='bn5')
        pool5 = tl.layers.PoolLayer(conv5, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                    padding='SAME', pool = tf.nn.max_pool, name ='pool5')

        
        fc6 = tl.layers.Conv2d(pool5, 256, (7, 7), act=tf.nn.relu, name='fc6')
        #================drop=======================================
        drop6 = tl.layers.DropoutLayer(fc6, keep=0.8, name='drop6',is_train=is_train)
        
        fc7 = tl.layers.Conv2d(drop6, 256, (1, 1), act=tf.nn.relu, name='fc7')
        #================drop=======================================
        drop7 = tl.layers.DropoutLayer(fc7, keep=0.8, name='drop7',is_train=is_train)
        
        score_fr = tl.layers.Conv2d(drop7 , 2, (1, 1), name='score_fr')
        upscore2 = tl.layers.DeConv2d(score_fr, 2, (4, 4), (nx/16, ny/16), (2, 2), name='upscore2')
        score_pool4 = tl.layers.Conv2d(pool4, 2, (1, 1), name='score_pool4') 
        fuse_pool4 = tl.layers.ElementwiseLayer([upscore2, score_pool4], combine_fn = tf.add, name='fuse_pool4')
        
        upscore_pool4 = tl.layers.DeConv2d(fuse_pool4, 2, (4, 4), (nx/8, ny/8), (2, 2), name='upscore_pool4')
        score_pool3 = tl.layers.Conv2d(pool3, 2, (1, 1), name='score_pool3') 
        fuse_pool3 = tl.layers.ElementwiseLayer([upscore_pool4, score_pool3],  combine_fn = tf.add, name='fuse_pool3')
        
        network = tl.layers.DeConv2d(fuse_pool3, 2, (16, 16), (nx, ny), (8, 8), name='upscore8')
        
        if is_train:
            #================Groundtruth==========================
            y_ = tf.reshape(y_,[batch_size*nx*ny,2])
            fw = tf.reshape(fw,[batch_size*nx*ny])
            
            y = network.outputs   
            y = tf.reshape(y,[batch_size*nx*ny,2])
            y_prob = tf.nn.softmax(y)
            y_class = tf.argmax(y_prob, 1)
            y_class = tf.reshape(y_class,[batch_size,nx,ny,1])           
            cost = -tf.reduce_mean(tf.multiply(tf.reduce_sum(y_*tf.log(y_prob),1),fw))
            return network,cost,y_class 
        else:
            y = network.outputs   
            y = tf.reshape(y,[nx*ny,2])
            y_prob = tf.nn.softmax(y)
            y_class = tf.argmax(y_prob, 1)
            y_prob = tf.reshape(y_prob,[batch_size, data_shape[0],data_shape[1],2])
            y_prob = tf.slice(y_prob,[0,0,0,1],[batch_size, data_shape[0],data_shape[1],1])
            y_class = tf.reshape(y_class,[batch_size,nx,ny,1])
            return network,y_class, y_prob 

def model_VGG16_HED(x, y_,fw, batch_size, data_shape, reuse=False, mean_file_name=None,is_train = True, network_scopename = "VGG16_HED" ):
    
    if mean_file_name!=None:
        meanval = np.load(mean_file_name)
        x = image_preprocess(x,meanval)
    
    nx = data_shape[0]
    ny = data_shape[1]
  
    gamma_init=tf.random_normal_initializer(2., 0.1)
    with tf.variable_scope(network_scopename, reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        network_input = tl.layers.InputLayer(x, name='input')
        """ conv1 """
        conv1 = tl.layers.Conv2dLayer(network_input, shape = [3, 3, 3, 64],
                    W_init=tf.contrib.layers.xavier_initializer_conv2d(uniform=True, seed=None, dtype=tf.float32),
                    b_init = tf.constant_initializer(value=0.0),
                    strides = [1, 1, 1, 1], padding='SAME', name ='conv1_1')
        conv1 = tl.layers.Conv2dLayer(conv1, shape = [3, 3, 64, 64],    # 64 features for each 3x3 patch
                    W_init=tf.contrib.layers.xavier_initializer_conv2d(uniform=True, seed=None, dtype=tf.float32),
                    b_init = tf.constant_initializer(value=0.0),
                    strides = [1, 1, 1, 1], padding='SAME', name ='conv1_2')
        conv1 = tl.layers.BatchNormLayer(conv1, act = tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='bn1')
        pool1 = tl.layers.PoolLayer(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],padding='SAME', 
                    pool = tf.nn.max_pool, name ='pool1') #outputsize: [H/2,W/2]
        """ conv2 """
        conv2 = tl.layers.Conv2dLayer(pool1, shape = [3, 3, 64, 128],  # 128 features for each 3x3 patch
                    W_init=tf.contrib.layers.xavier_initializer_conv2d(uniform=True, seed=None, dtype=tf.float32),
                    b_init = tf.constant_initializer(value=0.0),
                                         strides = [1, 1, 1, 1], padding='SAME', name ='conv2_1')
        conv2 = tl.layers.Conv2dLayer(conv2, shape = [3, 3, 128, 128],  # 128 features for each 3x3 patch
                    W_init=tf.contrib.layers.xavier_initializer_conv2d(uniform=True, seed=None, dtype=tf.float32),
                    b_init = tf.constant_initializer(value=0.0),
                                         strides = [1, 1, 1, 1], padding='SAME', name ='conv2_2')
        conv2 = tl.layers.BatchNormLayer(conv2, act = tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='bn2')
        pool2 = tl.layers.PoolLayer(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                    padding='SAME', pool = tf.nn.max_pool, name ='pool2') #outputsize: [H/4,W/4]
        """ conv3 """
        conv3 = tl.layers.Conv2dLayer(pool2, shape = [3, 3, 128, 256],  # 256 features for each 3x3 patch
                    W_init=tf.contrib.layers.xavier_initializer_conv2d(uniform=True, seed=None, dtype=tf.float32),
                    b_init = tf.constant_initializer(value=0.0),
                                         strides = [1, 1, 1, 1], padding='SAME', name ='conv3_1')
        conv3 = tl.layers.Conv2dLayer(conv3, shape = [3, 3, 256, 256],  # 256 features for each 3x3 patch
                    W_init=tf.contrib.layers.xavier_initializer_conv2d(uniform=True, seed=None, dtype=tf.float32),
                    b_init = tf.constant_initializer(value=0.0),
                                         strides = [1, 1, 1, 1], padding='SAME', name ='conv3_2')
        conv3 = tl.layers.Conv2dLayer(conv3, shape = [3, 3, 256, 256],  # 256 features for each 3x3 patch
                    W_init=tf.contrib.layers.xavier_initializer_conv2d(uniform=True, seed=None, dtype=tf.float32),
                    b_init = tf.constant_initializer(value=0.0),
                                         strides = [1, 1, 1, 1], padding='SAME', name ='conv3_3')
        conv3 = tl.layers.BatchNormLayer(conv3, act = tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='bn3')
        pool3 = tl.layers.PoolLayer(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                    padding='SAME', pool = tf.nn.max_pool, name ='pool3') #outputsize: [H/8,W/8]
        """ conv4 """
        conv4 = tl.layers.Conv2dLayer(pool3, shape = [3, 3, 256, 512],  # 512 features for each 3x3 patch
                    W_init=tf.contrib.layers.xavier_initializer_conv2d(uniform=True, seed=None, dtype=tf.float32),
                    b_init = tf.constant_initializer(value=0.0),
                                         strides = [1, 1, 1, 1], padding='SAME', name ='conv4_1')
        conv4 = tl.layers.Conv2dLayer(conv4, shape = [3, 3, 512, 512],  # 512 features for each 3x3 patch
                    W_init=tf.contrib.layers.xavier_initializer_conv2d(uniform=True, seed=None, dtype=tf.float32),
                    b_init = tf.constant_initializer(value=0.0),
                                         strides = [1, 1, 1, 1], padding='SAME', name ='conv4_2')
        conv4 = tl.layers.Conv2dLayer(conv4, shape = [3, 3, 512, 512],  # 512 features for each 3x3 patch
                    strides = [1, 1, 1, 1], padding='SAME', name ='conv4_3')
        conv4 = tl.layers.BatchNormLayer(conv4, act = tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='bn4')
        pool4 = tl.layers.PoolLayer(conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                    padding='SAME', pool = tf.nn.max_pool, name ='pool4') #outputsize: [H/16,W/16]
        """ conv5 """
        conv5 = tl.layers.Conv2dLayer(pool4, shape = [3, 3, 512, 512],  # 512 features for each 3x3 patch
                    W_init=tf.contrib.layers.xavier_initializer_conv2d(uniform=True, seed=None, dtype=tf.float32),
                    b_init = tf.constant_initializer(value=0.0),
                                         strides = [1, 1, 1, 1], padding='SAME', name ='conv5_1')
        conv5 = tl.layers.Conv2dLayer(conv5, shape = [3, 3, 512, 512],  # 512 features for each 3x3 patch
                    W_init=tf.contrib.layers.xavier_initializer_conv2d(uniform=True, seed=None, dtype=tf.float32),
                    b_init = tf.constant_initializer(value=0.0),
                                         strides = [1, 1, 1, 1], padding='SAME', name ='conv5_2')
        conv5 = tl.layers.Conv2dLayer(conv5, shape = [3, 3, 512, 512],  # 512 features for each 3x3 patch
                    W_init=tf.contrib.layers.xavier_initializer_conv2d(uniform=True, seed=None, dtype=tf.float32),
                    b_init = tf.constant_initializer(value=0.0),
                                         strides = [1, 1, 1, 1], padding='SAME', name ='conv5_3')
        conv5 = tl.layers.BatchNormLayer(conv5, act = tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='bn5')
        

        score_dsn1 = tl.layers.Conv2d(conv1, 2, (1, 1), name='score_dsn1')
        
        score_dsn2 = tl.layers.Conv2d(conv2, 1, (1, 1), name='score_dsn2')
        upsample_2 = tl.layers.DeConv2d(score_dsn2, 2, (4, 4), (nx, ny), (2, 2), name='upsample_2')
        
        score_dsn3 = tl.layers.Conv2d(conv3, 1, (1, 1), name='score_dsn3')
        upsample_3 = tl.layers.DeConv2d(score_dsn3, 2, (8, 8), (nx, ny), (4, 4), name='upsample_3')
        
        score_dsn4 = tl.layers.Conv2d(conv4, 1, (1, 1), name='score_dsn4')
        upsample_4 = tl.layers.DeConv2d(score_dsn4, 2, (16, 16), (nx, ny), (8, 8), name='upsample_4')
        
        score_dsn5 = tl.layers.Conv2d(conv5, 1, (1, 1), name='score_dsn5')
        upsample_5 = tl.layers.DeConv2d(score_dsn5, 2, (32, 32), (nx, ny), (16, 16), name='upsample_5')
        
        concatall = tl.layers.ConcatLayer([score_dsn1,upsample_2,upsample_3,upsample_4,upsample_5], 3, name='concatall')
        network = tl.layers.Conv2d(concatall, 2, (1, 1), name='output')
        
        if is_train:
            #================Groundtruth==========================
            y_ = tf.reshape(y_,[batch_size*nx*ny,2])
            fw = tf.reshape(fw,[batch_size*nx*ny])
            
            y1 = score_dsn1.outputs   
            y1 = tf.reshape(y1,[batch_size*nx*ny,2])
            y1_prob = tf.nn.softmax(y1)        
            cost1 = -tf.reduce_mean(tf.multiply(tf.reduce_sum(y_*tf.log(y1_prob),1),fw))
            
            y2 = upsample_2.outputs   
            y2 = tf.reshape(y2,[batch_size*nx*ny,2])
            y2_prob = tf.nn.softmax(y2)        
            cost2 = -tf.reduce_mean(tf.multiply(tf.reduce_sum(y_*tf.log(y2_prob),1),fw))
            
            y3 = upsample_3.outputs   
            y3 = tf.reshape(y3,[batch_size*nx*ny,2])
            y3_prob = tf.nn.softmax(y3)          
            cost3 = -tf.reduce_mean(tf.multiply(tf.reduce_sum(y_*tf.log(y3_prob),1),fw))
            
            y4 = upsample_4.outputs   
            y4 = tf.reshape(y4,[batch_size*nx*ny,2])
            y4_prob = tf.nn.softmax(y4)          
            cost4 = -tf.reduce_mean(tf.multiply(tf.reduce_sum(y_*tf.log(y4_prob),1),fw))
            
            y5 = upsample_5.outputs   
            y5 = tf.reshape(y5,[batch_size*nx*ny,2])
            y5_prob = tf.nn.softmax(y5)          
            cost5 = -tf.reduce_mean(tf.multiply(tf.reduce_sum(y_*tf.log(y5_prob),1),fw))
            
            y = network.outputs   
            y = tf.reshape(y,[batch_size*nx*ny,2])
            y_prob = tf.nn.softmax(y)
            y_class = tf.argmax(y_prob, 1)
            y_class = tf.reshape(y_class,[batch_size,nx,ny,1])           
            cost_fuse = -tf.reduce_mean(tf.multiply(tf.reduce_sum(y_*tf.log(y_prob),1),fw))
            
            cost = cost1+cost2+cost3+cost4+cost5+5*cost_fuse
        
            return network,cost,y_class 
        else:
            y = network.outputs   
            y = tf.reshape(y,[nx*ny,2])
            y_prob = tf.nn.softmax(y)
            y_class = tf.argmax(y_prob, 1)
            y_prob = tf.reshape(y_prob,[batch_size, data_shape[0],data_shape[1],2])
            y_prob = tf.slice(y_prob,[0,0,0,1],[batch_size, data_shape[0],data_shape[1],1])
            y_class = tf.reshape(y_class,[batch_size,nx,ny,1])
            return network,y_class, y_prob 
            
            
def model_VGG16_Unet(x, y_,fw, batch_size, data_shape, reuse=False, mean_file_name=None,is_train = True, network_scopename = "VGG16_Unet" ):
    
    if mean_file_name!=None:
        meanval = np.load(mean_file_name)
        x = image_preprocess(x,meanval)
    
    nx = data_shape[0]
    ny = data_shape[1]
  
    gamma_init=tf.random_normal_initializer(2., 0.1)
    with tf.variable_scope(network_scopename, reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        network_input = tl.layers.InputLayer(x, name='input')
        """ conv1 """
        conv1 = tl.layers.Conv2dLayer(network_input, shape = [3, 3, 3, 64],
                    W_init=tf.contrib.layers.xavier_initializer_conv2d(uniform=True, seed=None, dtype=tf.float32),
                    b_init = tf.constant_initializer(value=0.0),
                    strides = [1, 1, 1, 1], padding='SAME', name ='conv1_1')
        conv1 = tl.layers.Conv2dLayer(conv1, shape = [3, 3, 64, 64],    # 64 features for each 3x3 patch
                    W_init=tf.contrib.layers.xavier_initializer_conv2d(uniform=True, seed=None, dtype=tf.float32),
                    b_init = tf.constant_initializer(value=0.0),
                    strides = [1, 1, 1, 1], padding='SAME', name ='conv1_2')
        conv1 = tl.layers.BatchNormLayer(conv1, act = tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='bn1')
        pool1 = tl.layers.PoolLayer(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],padding='SAME', 
                    pool = tf.nn.max_pool, name ='pool1') #outputsize: [H/2,W/2]
        """ conv2 """
        conv2 = tl.layers.Conv2dLayer(pool1, shape = [3, 3, 64, 128],  # 128 features for each 3x3 patch
                    W_init=tf.contrib.layers.xavier_initializer_conv2d(uniform=True, seed=None, dtype=tf.float32),
                    b_init = tf.constant_initializer(value=0.0),
                                         strides = [1, 1, 1, 1], padding='SAME', name ='conv2_1')
        conv2 = tl.layers.Conv2dLayer(conv2, shape = [3, 3, 128, 128],  # 128 features for each 3x3 patch
                    W_init=tf.contrib.layers.xavier_initializer_conv2d(uniform=True, seed=None, dtype=tf.float32),
                    b_init = tf.constant_initializer(value=0.0),
                                         strides = [1, 1, 1, 1], padding='SAME', name ='conv2_2')
        conv2 = tl.layers.BatchNormLayer(conv2, act = tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='bn2')
        pool2 = tl.layers.PoolLayer(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                    padding='SAME', pool = tf.nn.max_pool, name ='pool2') #outputsize: [H/4,W/4]
        """ conv3 """
        conv3 = tl.layers.Conv2dLayer(pool2, shape = [3, 3, 128, 256],  # 256 features for each 3x3 patch
                    W_init=tf.contrib.layers.xavier_initializer_conv2d(uniform=True, seed=None, dtype=tf.float32),
                    b_init = tf.constant_initializer(value=0.0),
                                         strides = [1, 1, 1, 1], padding='SAME', name ='conv3_1')
        conv3 = tl.layers.Conv2dLayer(conv3, shape = [3, 3, 256, 256],  # 256 features for each 3x3 patch
                    W_init=tf.contrib.layers.xavier_initializer_conv2d(uniform=True, seed=None, dtype=tf.float32),
                    b_init = tf.constant_initializer(value=0.0),
                                         strides = [1, 1, 1, 1], padding='SAME', name ='conv3_2')
        conv3 = tl.layers.Conv2dLayer(conv3, shape = [3, 3, 256, 256],  # 256 features for each 3x3 patch
                    W_init=tf.contrib.layers.xavier_initializer_conv2d(uniform=True, seed=None, dtype=tf.float32),
                    b_init = tf.constant_initializer(value=0.0),
                                         strides = [1, 1, 1, 1], padding='SAME', name ='conv3_3')
        conv3 = tl.layers.BatchNormLayer(conv3, act = tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='bn3')
        pool3 = tl.layers.PoolLayer(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                    padding='SAME', pool = tf.nn.max_pool, name ='pool3') #outputsize: [H/8,W/8]
        """ conv4 """
        conv4 = tl.layers.Conv2dLayer(pool3, shape = [3, 3, 256, 512],  # 512 features for each 3x3 patch
                    W_init=tf.contrib.layers.xavier_initializer_conv2d(uniform=True, seed=None, dtype=tf.float32),
                    b_init = tf.constant_initializer(value=0.0),
                                         strides = [1, 1, 1, 1], padding='SAME', name ='conv4_1')
        conv4 = tl.layers.Conv2dLayer(conv4, shape = [3, 3, 512, 512],  # 512 features for each 3x3 patch
                    W_init=tf.contrib.layers.xavier_initializer_conv2d(uniform=True, seed=None, dtype=tf.float32),
                    b_init = tf.constant_initializer(value=0.0),
                                         strides = [1, 1, 1, 1], padding='SAME', name ='conv4_2')
        conv4 = tl.layers.Conv2dLayer(conv4, shape = [3, 3, 512, 512],  # 512 features for each 3x3 patch
                    strides = [1, 1, 1, 1], padding='SAME', name ='conv4_3')
        conv4 = tl.layers.BatchNormLayer(conv4, act = tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='bn4')
        pool4 = tl.layers.PoolLayer(conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                    padding='SAME', pool = tf.nn.max_pool, name ='pool4') #outputsize: [H/16,W/16]
        """ conv5 """
        conv5 = tl.layers.Conv2dLayer(pool4, shape = [3, 3, 512, 512],  # 512 features for each 3x3 patch
                    W_init=tf.contrib.layers.xavier_initializer_conv2d(uniform=True, seed=None, dtype=tf.float32),
                    b_init = tf.constant_initializer(value=0.0),
                                         strides = [1, 1, 1, 1], padding='SAME', name ='conv5_1')
        conv5 = tl.layers.Conv2dLayer(conv5, shape = [3, 3, 512, 512],  # 512 features for each 3x3 patch
                    W_init=tf.contrib.layers.xavier_initializer_conv2d(uniform=True, seed=None, dtype=tf.float32),
                    b_init = tf.constant_initializer(value=0.0),
                                         strides = [1, 1, 1, 1], padding='SAME', name ='conv5_2')
        conv5 = tl.layers.Conv2dLayer(conv5, shape = [3, 3, 512, 512],  # 512 features for each 3x3 patch
                    W_init=tf.contrib.layers.xavier_initializer_conv2d(uniform=True, seed=None, dtype=tf.float32),
                    b_init = tf.constant_initializer(value=0.0),
                                         strides = [1, 1, 1, 1], padding='SAME', name ='conv5_3')
        conv5 = tl.layers.BatchNormLayer(conv5, act = tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='bn5')
        
        up4 = tl.layers.DeConv2d(conv5, 256, (3, 3), (nx/8, ny/8), (2, 2), name='deconv4')
        up4 = tl.layers.ConcatLayer([up4, conv4], 3, name='concat4')
        conv4 = tl.layers.Conv2d(up4, 256, (3, 3), act=tf.nn.relu, name='uconv4_1')
        conv4 = tl.layers.Conv2d(conv4, 256, (3, 3), act=tf.nn.relu, name='uconv4_2')
        up3 = tl.layers.DeConv2d(conv4, 128, (3, 3), (nx/4, ny/4), (2, 2), name='deconv3')
        up3 = tl.layers.ConcatLayer([up3, conv3], 3, name='concat3')
        conv3 = tl.layers.Conv2d(up3, 128, (3, 3), act=tf.nn.relu, name='uconv3_1')
        conv3 = tl.layers.Conv2d(conv3, 128, (3, 3), act=tf.nn.relu, name='uconv3_2')
        up2 = tl.layers.DeConv2d(conv3, 64, (3, 3), (nx/2, ny/2), (2, 2), name='deconv2')
        up2 = tl.layers.ConcatLayer([up2, conv2], 3, name='concat2')
        conv2 = tl.layers.Conv2d(up2, 64, (3, 3), act=tf.nn.relu,  name='uconv2_1')
        conv2 = tl.layers.Conv2d(conv2, 64, (3, 3), act=tf.nn.relu, name='uconv2_2')
        up1 = tl.layers.DeConv2d(conv2, 32, (3, 3), (nx/1, ny/1), (2, 2), name='deconv1')
        up1 = tl.layers.ConcatLayer([up1, conv1] , 3, name='concat1')
        conv1 = tl.layers.Conv2d(up1, 64, (3, 3), act=tf.nn.relu, name='uconv1_1')
        conv1 = tl.layers.Conv2d(conv1, 64, (3, 3), act=tf.nn.relu, name='uconv1_2')
        network = tl.layers.Conv2d(conv1, 2, (1, 1), act=tf.nn.sigmoid, name='uconv1')
        
        if is_train:
            #================Groundtruth==========================
            y_ = tf.reshape(y_,[batch_size*nx*ny,2])
            fw = tf.reshape(fw,[batch_size*nx*ny])
            
            y = network.outputs   
            y = tf.reshape(y,[batch_size*nx*ny,2])
            y_prob = tf.nn.softmax(y)
            y_class = tf.argmax(y_prob, 1)
            y_class = tf.reshape(y_class,[batch_size,nx,ny,1])           
            cost = -tf.reduce_mean(tf.multiply(tf.reduce_sum(y_*tf.log(y_prob),1),fw))
            return network,cost,y_class 
        else:
            y = network.outputs   
            y = tf.reshape(y,[nx*ny,2])
            y_prob = tf.nn.softmax(y)
            y_class = tf.argmax(y_prob, 1)
            y_prob = tf.reshape(y_prob,[batch_size, data_shape[0],data_shape[1],2])
            y_prob = tf.slice(y_prob,[0,0,0,1],[batch_size, data_shape[0],data_shape[1],1])
            y_class = tf.reshape(y_class,[batch_size,nx,ny,1])
            return network,y_class, y_prob   
            
def model_VGG16_SharpMask(x, y_,fw, batch_size, data_shape, reuse=False, mean_file_name=None,is_train = True, network_scopename = "VGG16_SharpMask" ):
    
    if mean_file_name!=None:
        meanval = np.load(mean_file_name)
        x = image_preprocess(x,meanval)
    nx = data_shape[0]
    ny = data_shape[1]
    gamma_init=tf.random_normal_initializer(2., 0.1)
    with tf.variable_scope(network_scopename, reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        network_input = tl.layers.InputLayer(x, name='input')
        """ conv1 """
        conv1 = tl.layers.Conv2dLayer(network_input, shape = [3, 3, 3, 64],
                    W_init=tf.contrib.layers.xavier_initializer_conv2d(uniform=True, seed=None, dtype=tf.float32),
                    b_init = tf.constant_initializer(value=0.0),
                    strides = [1, 1, 1, 1], padding='SAME', name ='conv1_1')
        conv1 = tl.layers.Conv2dLayer(conv1, shape = [3, 3, 64, 64],    # 64 features for each 3x3 patch
                    W_init=tf.contrib.layers.xavier_initializer_conv2d(uniform=True, seed=None, dtype=tf.float32),
                    b_init = tf.constant_initializer(value=0.0),
                    strides = [1, 1, 1, 1], padding='SAME', name ='conv1_2')
        conv1 = tl.layers.BatchNormLayer(conv1, act = tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='bn1')
        pool1 = tl.layers.PoolLayer(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],padding='SAME', 
                    pool = tf.nn.max_pool, name ='pool1') #outputsize: [H/2,W/2]
        """ conv2 """
        conv2 = tl.layers.Conv2dLayer(pool1, shape = [3, 3, 64, 128],  # 128 features for each 3x3 patch
                    W_init=tf.contrib.layers.xavier_initializer_conv2d(uniform=True, seed=None, dtype=tf.float32),
                    b_init = tf.constant_initializer(value=0.0),
                                         strides = [1, 1, 1, 1], padding='SAME', name ='conv2_1')
        conv2 = tl.layers.Conv2dLayer(conv2, shape = [3, 3, 128, 128],  # 128 features for each 3x3 patch
                    W_init=tf.contrib.layers.xavier_initializer_conv2d(uniform=True, seed=None, dtype=tf.float32),
                    b_init = tf.constant_initializer(value=0.0),
                                         strides = [1, 1, 1, 1], padding='SAME', name ='conv2_2')
        conv2 = tl.layers.BatchNormLayer(conv2, act = tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='bn2')
        pool2 = tl.layers.PoolLayer(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                    padding='SAME', pool = tf.nn.max_pool, name ='pool2') #outputsize: [H/4,W/4]
        """ conv3 """
        conv3 = tl.layers.Conv2dLayer(pool2, shape = [3, 3, 128, 256],  # 256 features for each 3x3 patch
                    W_init=tf.contrib.layers.xavier_initializer_conv2d(uniform=True, seed=None, dtype=tf.float32),
                    b_init = tf.constant_initializer(value=0.0),
                                         strides = [1, 1, 1, 1], padding='SAME', name ='conv3_1')
        conv3 = tl.layers.Conv2dLayer(conv3, shape = [3, 3, 256, 256],  # 256 features for each 3x3 patch
                    W_init=tf.contrib.layers.xavier_initializer_conv2d(uniform=True, seed=None, dtype=tf.float32),
                    b_init = tf.constant_initializer(value=0.0),
                                         strides = [1, 1, 1, 1], padding='SAME', name ='conv3_2')
        conv3 = tl.layers.Conv2dLayer(conv3, shape = [3, 3, 256, 256],  # 256 features for each 3x3 patch
                    W_init=tf.contrib.layers.xavier_initializer_conv2d(uniform=True, seed=None, dtype=tf.float32),
                    b_init = tf.constant_initializer(value=0.0),
                                         strides = [1, 1, 1, 1], padding='SAME', name ='conv3_3')
        conv3 = tl.layers.BatchNormLayer(conv3, act = tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='bn3')
        pool3 = tl.layers.PoolLayer(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                    padding='SAME', pool = tf.nn.max_pool, name ='pool3') #outputsize: [H/8,W/8]
        """ conv4 """
        conv4 = tl.layers.Conv2dLayer(pool3, shape = [3, 3, 256, 512],  # 512 features for each 3x3 patch
                    W_init=tf.contrib.layers.xavier_initializer_conv2d(uniform=True, seed=None, dtype=tf.float32),
                    b_init = tf.constant_initializer(value=0.0),
                                         strides = [1, 1, 1, 1], padding='SAME', name ='conv4_1')
        conv4 = tl.layers.Conv2dLayer(conv4, shape = [3, 3, 512, 512],  # 512 features for each 3x3 patch
                    W_init=tf.contrib.layers.xavier_initializer_conv2d(uniform=True, seed=None, dtype=tf.float32),
                    b_init = tf.constant_initializer(value=0.0),
                                         strides = [1, 1, 1, 1], padding='SAME', name ='conv4_2')
        conv4 = tl.layers.Conv2dLayer(conv4, shape = [3, 3, 512, 512],  # 512 features for each 3x3 patch
                    strides = [1, 1, 1, 1], padding='SAME', name ='conv4_3')
        conv4 = tl.layers.BatchNormLayer(conv4, act = tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='bn4')
        pool4 = tl.layers.PoolLayer(conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                    padding='SAME', pool = tf.nn.max_pool, name ='pool4') #outputsize: [H/16,W/16]
        #============================================================================
        conv5 =  tl.layers.Conv2dLayer(pool4, shape = [1, 1, 512, 256],  # 512 features for each 3x3 patch
                    W_init=tf.contrib.layers.xavier_initializer_conv2d(uniform=True, seed=None, dtype=tf.float32),
                    b_init = tf.constant_initializer(value=0.0),
                                         strides = [1, 1, 1, 1], padding='SAME', name ='conv5')  
        #==============refine 1=======================
        conv4_refine = tl.layers.Conv2dLayer(pool4, shape = [1, 1, 512, 256],  # 512 features for each 3x3 patch
                    W_init=tf.contrib.layers.xavier_initializer_conv2d(uniform=True, seed=None, dtype=tf.float32),
                    b_init = tf.constant_initializer(value=0.0),strides = [1, 1, 1, 1], padding='SAME', name ='conv4_refine')          
        refine1_cmb = tl.layers.ConcatLayer([conv4_refine,conv5],
                    concat_dim = 3, name = 'refine1_cmb')# output:[H/8,W/8,192]
        refine1 = tl.layers.Conv2dLayer(refine1_cmb, shape = [3, 3, 512, 128],  # 512 features for each 3x3 patch
                    W_init=tf.contrib.layers.xavier_initializer_conv2d(uniform=True, seed=None, dtype=tf.float32),
                    b_init = tf.constant_initializer(value=0.0),strides = [1, 1, 1, 1], padding='SAME', name ='refine1')  
        refine1 = tl.layers.UpSampling2dLayer(refine1, 
                    size = [data_shape[0]//8,data_shape[1]//8], method =0,is_scale = False,name = 'refine1_up' )   
        #==============refine 2=======================   
        conv3_refine = tl.layers.Conv2dLayer(pool3, shape = [1, 1, 256, 128],  # 512 features for each 3x3 patch
                    W_init=tf.contrib.layers.xavier_initializer_conv2d(uniform=True, seed=None, dtype=tf.float32),
                    b_init = tf.constant_initializer(value=0.0),strides = [1, 1, 1, 1], padding='SAME', name ='conv3_refine')          
        refine2_cmb = tl.layers.ConcatLayer([conv3_refine,refine1],
                    concat_dim = 3, name = 'refine2_cmb')# output:[H/8,W/8,192]
        refine2 = tl.layers.Conv2dLayer(refine2_cmb, shape = [3, 3, 256, 64],  # 512 features for each 3x3 patch
                    W_init=tf.contrib.layers.xavier_initializer_conv2d(uniform=True, seed=None, dtype=tf.float32),
                    b_init = tf.constant_initializer(value=0.0),strides = [1, 1, 1, 1], padding='SAME', name ='refine2')  
        refine2 = tl.layers.UpSampling2dLayer(refine2, 
                    size = [data_shape[0]//4,data_shape[1]//4], method =0,is_scale = False,name = 'refine2_up' )   
        #==============refine 3=======================   
        conv2_refine = tl.layers.Conv2dLayer(pool2, shape = [1, 1, 128, 64],  # 512 features for each 3x3 patch
                    W_init=tf.contrib.layers.xavier_initializer_conv2d(uniform=True, seed=None, dtype=tf.float32),
                    b_init = tf.constant_initializer(value=0.0),strides = [1, 1, 1, 1], padding='SAME', name ='conv2_refine')          
        refine3_cmb = tl.layers.ConcatLayer([conv2_refine,refine2],
                    concat_dim = 3, name = 'refine3_cmb')# output:[H/8,W/8,192]
        refine3 = tl.layers.Conv2dLayer(refine3_cmb, shape = [3, 3, 128, 64],  # 512 features for each 3x3 patch
                    W_init=tf.contrib.layers.xavier_initializer_conv2d(uniform=True, seed=None, dtype=tf.float32),
                    b_init = tf.constant_initializer(value=0.0),strides = [1, 1, 1, 1], padding='SAME', name ='refine3')
        refine3 = tl.layers.UpSampling2dLayer(refine3, 
                    size = [data_shape[0]//2,data_shape[1]//2], method =0,is_scale = False,name = 'refine3_up' )   
        #==============refine 4=======================   
        conv1_refine = tl.layers.Conv2dLayer(pool1, shape = [1, 1, 64, 64],  # 512 features for each 3x3 patch
                    W_init=tf.contrib.layers.xavier_initializer_conv2d(uniform=True, seed=None, dtype=tf.float32),
                    b_init = tf.constant_initializer(value=0.0),strides = [1, 1, 1, 1], padding='SAME', name ='conv1_refine')          
        refine4_cmb = tl.layers.ConcatLayer([conv1_refine,refine3],
                    concat_dim = 3, name = 'refine4_cmb')# output:[H/8,W/8,192]
        refine4 = tl.layers.Conv2dLayer(refine4_cmb, shape = [3, 3, 128, 128],  # 512 features for each 3x3 patch
                    W_init=tf.contrib.layers.xavier_initializer_conv2d(uniform=True, seed=None, dtype=tf.float32),
                    b_init = tf.constant_initializer(value=0.0),strides = [1, 1, 1, 1], padding='SAME', name ='refine4') 
        refine4 = tl.layers.UpSampling2dLayer(refine4, 
                    size = [data_shape[0],data_shape[1]], method =0,is_scale = False,name = 'refine4_up' )   
        network = tl.layers.Conv2d(refine4, 2, (3, 3), name='score')
        
        if is_train:
            #================Groundtruth==========================
            y_ = tf.reshape(y_,[batch_size*nx*ny,2])
            fw = tf.reshape(fw,[batch_size*nx*ny])
            
            y = network.outputs   
            y = tf.reshape(y,[batch_size*nx*ny,2])
            y_prob = tf.nn.softmax(y)
            y_class = tf.argmax(y_prob, 1)
            y_class = tf.reshape(y_class,[batch_size,nx,ny,1])           
            cost = -tf.reduce_mean(tf.multiply(tf.reduce_sum(y_*tf.log(y_prob),1),fw))
            return network,cost,y_class 
        else:
            y = network.outputs   
            y = tf.reshape(y,[nx*ny,2])
            y_prob = tf.nn.softmax(y)
            y_class = tf.argmax(y_prob, 1)
            y_prob = tf.reshape(y_prob,[batch_size, data_shape[0],data_shape[1],2])
            y_prob = tf.slice(y_prob,[0,0,0,1],[batch_size, data_shape[0],data_shape[1],1])
            y_class = tf.reshape(y_class,[batch_size,nx,ny,1])
            return network,y_class, y_prob   
            

