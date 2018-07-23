# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 21:03:21 2018

@author: Xie Lipeng
"""

import numpy as np
from utils.write_read_tfrecord import *
import tensorflow as tf
import tensorlayer as tl
import time
from nets.construct_model_multiple_networks import *
import os
import matplotlib.pyplot as plt

#=========================step1: Set parameters===============================
DatasetName ="MICCAI2017" # "MICCAI2017" or "BNS"
NetworkList = ['FCN8s','HED','Unet','SharpMask','DIMAN'] #'FCN8s','HED','Unet','SharpMask','DIMAN'
batch_size = 12 
n_epoch = 300 # number of iter  
learning_rate = 0.0002
option_savenet = True #save the parameters of network or not
data_shape = [224,224,3]
print_freq = 10 # the frequency of saving parameters
c = [1,1,1] # the weight of three tasks in DIMAN

#=======================step2: Creat folder and Load data=====================
datapath = './data/'+ DatasetName+ '/'  #the path to store patches and lmdb
train_tfrecord = datapath +'train_1_1.tfrecords'
mean_file_name = datapath +'train_1_1.tfrecords_mean.npy'

trainnum = sum(1 for _ in tf.python_io.tf_record_iterator(train_tfrecord)) #计算训练数据数量
min_after_dequeue_train = trainnum
capacity_train = min_after_dequeue_train + 3 * batch_size

n_step_epoch = int(trainnum/batch_size)
n_step = n_epoch * n_step_epoch
#-------------------------------Load Dat----------------------------------
train_img, train_label, train_foregroundweight, train_intervalmask, train_intervalweight, train_maskermask, train_maskerweight= read_and_decode(train_tfrecord,data_shape)      
X_train, y_train, fw_train, im_train, iw_train, mm_train, mw_train = tf.train.shuffle_batch([train_img, train_label, train_foregroundweight, train_intervalmask, train_intervalweight, train_maskermask, train_maskerweight],
                                                batch_size=batch_size, capacity=1000,
                                                min_after_dequeue=500)
#------------------------------------------------------------------------------
x = tf.placeholder(tf.float32, shape=[batch_size, data_shape[0],data_shape[1], 3])   # [batch_size, height, width, channels]
y_ = tf.placeholder(tf.float32, shape=[batch_size,data_shape[0],data_shape[1], 2])
fw = tf.placeholder(tf.float32, shape=[batch_size,data_shape[0],data_shape[1]])
im = tf.placeholder(tf.float32, shape=[batch_size, data_shape[0],data_shape[1], 2])   # [batch_size, height, width, channels]
iw = tf.placeholder(tf.float32, shape=[batch_size,data_shape[0],data_shape[1]])
mm = tf.placeholder(tf.float32, shape=[batch_size, data_shape[0],data_shape[1], 2])   # [batch_size, height, width, channels]
mw = tf.placeholder(tf.float32, shape=[batch_size,data_shape[0],data_shape[1]])
lr = tf.placeholder(tf.float32, shape=[])

Traintime = []
#==================================step3:Train network=======================================
for NetworkName in NetworkList:
    model_path = './checkpoints/'+ DatasetName+ '/' + NetworkName + '/'
    if not os.path.exists(model_path):
        os.mkdir(model_path)
        
    network_scopename = NetworkName
    sess = tf.Session()
    coord = tf.train.Coordinator() #stop thread
    threads = tf.train.start_queue_runners(sess=sess,coord=coord)
    start_time = time.time()
    if NetworkName == 'DIMAN' or NetworkName == 'Unet_Multitask':
        network,cost,op_class_fg,op_class_interval,op_class_masker,op_class_refine = eval('model_VGG16_' + NetworkName + '(x,y_,fw,im,iw,mm,mw,c,batch_size,data_shape,reuse=False,mean_file_name=mean_file_name, is_train = True,network_scopename=network_scopename)')
        train_params = network.all_params
        train_op = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.8, beta2=0.999,
                                          epsilon=1e-06, use_locking=False).minimize(cost, var_list=train_params)
        init=tf.global_variables_initializer()  
        sess.run(init)
        network.print_params(False)
        network.print_layers()
        print('Start Training: %s' % NetworkName)

        step = 0
        for epoch in range(n_epoch):
            start_time = time.time()
            train_loss,  n_batch = 0, 0
            for s in range(n_step_epoch):
                X_train_a, y_train_a, fw_train_a, im_train_a, iw_train_a, mm_train_a, mw_train_a= sess.run([X_train, y_train,fw_train, im_train, iw_train, mm_train, mw_train])
                feed_dict = {x: X_train_a, y_: y_train_a,fw:fw_train_a,im: im_train_a, iw:iw_train_a, mm:mm_train_a, mw:mw_train_a, lr:learning_rate}
                feed_dict.update(network.all_drop)   # enable noise layers
                _, err, output_class_fg,output_class_interval,output_class_masker, output_class_refine = sess.run([train_op,cost,op_class_fg,
                                        op_class_interval,op_class_masker,op_class_refine], feed_dict=feed_dict)
                step += 1; train_loss += err; n_batch += 1        
            if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:
                print("Epoch %d : Step %d-%d of %d took %fs" % (epoch, step, step + n_step_epoch, n_step, time.time() - start_time))
                print("   train loss: %f" % (train_loss/ n_batch))
                learning_rate = np.maximum(learning_rate*0.8,0.00001) 
                if option_savenet:
                    print("Save model " + "!"*10)
                    saver = tf.train.Saver()
                    save_path = saver.save(sess, model_path+"model_parameters_"+str(epoch)+".ckpt")
        coord.request_stop() #stop thread
        coord.join(threads)
        sess.close()

    
    else:
        network,cost,op_class_fg = eval('model_VGG16_'+NetworkName+'(x,y_,fw,batch_size,data_shape,reuse=False,mean_file_name=mean_file_name, is_train = True,network_scopename=network_scopename)')
        train_params = network.all_params
        train_op = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.8, beta2=0.999,
                                          epsilon=1e-08, use_locking=False).minimize(cost, var_list=train_params)
        init=tf.global_variables_initializer()  
        sess.run(init)
        network.print_params(False)
        network.print_layers()
        print('Start Training: %s' % NetworkName)

        step = 0
        for epoch in range(n_epoch):
            start_time = time.time()
            train_loss,  n_batch = 0, 0
            for s in range(n_step_epoch):
                X_train_a, y_train_a, fw_train_a = sess.run([X_train, y_train,fw_train])
                feed_dict = {x: X_train_a, y_: y_train_a,fw:fw_train_a,lr:learning_rate}
                feed_dict.update(network.all_drop)   # enable noise layers
                _, err, output_class_fg = sess.run([train_op,cost,op_class_fg], feed_dict = feed_dict)
                step += 1; train_loss += err; n_batch += 1        
            if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:
                print("Epoch %d : Step %d-%d of %d took %fs" % (epoch, step, step + n_step_epoch, n_step, time.time() - start_time))
                print("   train loss: %f" % (train_loss/ n_batch))
                learning_rate = np.maximum(learning_rate*0.8,0.00001) 
                if option_savenet:
                    print("Save model " + "!"*10)
                    saver = tf.train.Saver()
                    save_path = saver.save(sess, model_path+"model_parameters_"+str(epoch)+".ckpt")
        coord.request_stop() #stop thread
        coord.join(threads)
        sess.close()
    totaltime = time.time() - start_time
    print(" Mean time: %f" % (totaltime))
    Traintime.append(totaltime)
















