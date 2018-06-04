# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 21:14:18 2017

@author: XLP
"""
import tensorflow as tf 
from PIL import Image
import numpy as np
import scipy.io as sio 

def write_images_tfrecord(imglistpath,imgpath,tfreconame,data_shape,opt_meanval=False):
    writer = tf.python_io.TFRecordWriter(tfreconame)
    # +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
    fp = open(imglistpath,"r")  
    lines = fp.readlines()#读取全部内容   
    N = len(lines)	# Number of data instances

    if opt_meanval:
        meanvalname = tfreconame+'_mean.npy'
        meanval = np.zeros(data_shape)
        for i in range(N):
            imgname = lines[i][0:-1]
            ##=========load and write image, label data, foreground_weight
            img = Image.open(imgpath + '/' + imgname + '.png')    
            label =  Image.open(imgpath + '/' + imgname + '_l.png')            
            ## sum all data
            meanval = meanval + np.array(img)
            img = img.tobytes()
            label = np.array(label.convert('L'),'uint8')//255
            label = label.tobytes()
            foreground_weight = sio.loadmat(imgpath + '/' + imgname + '_fw.mat')
            foreground_weight = foreground_weight['Foreground_weight']
            foreground_weight= foreground_weight.tobytes()
            ##========= load and write interval data=========
            inter_mask = Image.open(imgpath + '/' + imgname + '_im.png')
            inter_weight = sio.loadmat(imgpath + '/' + imgname + '_iw.mat')
            inter_mask = np.array(inter_mask.convert('L'),'uint8')//255
            inter_mask = inter_mask.tobytes()
            inter_weight = inter_weight['Interval_weight']
            inter_weight= inter_weight.tobytes()
            
            ##========= load and write masker data=========
            masker_mask = Image.open(imgpath + '/' + imgname + '_mm.png')
            masker_weight = sio.loadmat(imgpath + '/' + imgname + '_mw.mat')
            masker_mask = np.array(masker_mask.convert('L'),'uint8')//255
            masker_mask = masker_mask.tobytes()
            masker_weight = masker_weight['Masker_weight']
            masker_weight= masker_weight.tobytes()
           
           
            example = tf.train.Example(features=tf.train.Features(feature={
                'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img])), 
                'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[label])),
                'foreground_weight': tf.train.Feature(bytes_list=tf.train.BytesList(value=[foreground_weight])),
                'inter_mask': tf.train.Feature(bytes_list=tf.train.BytesList(value=[inter_mask])),
                'inter_weight': tf.train.Feature(bytes_list=tf.train.BytesList(value=[inter_weight])),   
                'masker_mask': tf.train.Feature(bytes_list=tf.train.BytesList(value=[masker_mask])),
                'masker_weight': tf.train.Feature(bytes_list=tf.train.BytesList(value=[masker_weight])),
            }))
            writer.write(example.SerializeToString())  #序列化为字符串 
        meanval = meanval / N
        meanval = np.mean(np.reshape(meanval,[data_shape[0]*data_shape[1],3]),0)
        np.save(meanvalname,meanval)
    else:
        for i in range(N):
            imgname = lines[i][0:-1]
            ##=========load and write image, label data, foreground_weight
            img = Image.open(imgpath + '/' + imgname + '.png')    
            label =  Image.open(imgpath + '/' + imgname + '_l.png')            
            img = img.tobytes()
            label = np.array(label.convert('L'),'uint8')//255
            label = label.tobytes()
            foreground_weight = sio.loadmat(imgpath + '/' + imgname + '_fw.mat')
            foreground_weight = foreground_weight['Foreground_weight']
            foreground_weight= foreground_weight.tobytes()
            ##========= load and write interval data=========
            inter_mask = Image.open(imgpath + '/' + imgname + '_im.png')
            inter_weight = sio.loadmat(imgpath + '/' + imgname + '_iw.mat')
            inter_mask = np.array(inter_mask.convert('L'),'uint8')//255
            inter_mask = inter_mask.tobytes()
            inter_weight = inter_weight['Interval_weight']
            inter_weight= inter_weight.tobytes()
            
            ##========= load and write masker data=========
            masker_mask = Image.open(imgpath + '/' + imgname + '_mm.png')
            masker_weight = sio.loadmat(imgpath + '/' + imgname + '_mw.mat')
            masker_mask = np.array(masker_mask.convert('L'),'uint8')//255
            masker_mask = masker_mask.tobytes()
            masker_weight = masker_weight['Masker_weight']
            masker_weight= masker_weight.tobytes()
           
           
            example = tf.train.Example(features=tf.train.Features(feature={
                'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img])), 
                'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[label])),
                'foreground_weight': tf.train.Feature(bytes_list=tf.train.BytesList(value=[foreground_weight])),
                'inter_mask': tf.train.Feature(bytes_list=tf.train.BytesList(value=[inter_mask])),
                'inter_weight': tf.train.Feature(bytes_list=tf.train.BytesList(value=[inter_weight])),   
                'masker_mask': tf.train.Feature(bytes_list=tf.train.BytesList(value=[masker_mask])),
                'masker_weight': tf.train.Feature(bytes_list=tf.train.BytesList(value=[masker_weight])),
            }))
            writer.write(example.SerializeToString())  #序列化为字符串 

    writer.close() 
    fp.close()

def read_and_decode(tfreconame,data_shape):
    #根据文件名生成一个队列
    filename_queue = tf.train.string_input_producer([tfreconame])

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)   #返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                      features={
                                           'image':tf.FixedLenFeature([],tf.string), 
                                           'label':tf.FixedLenFeature([],tf.string),
                                           'foreground_weight':tf.FixedLenFeature([],tf.string),
                                           'inter_mask':tf.FixedLenFeature([],tf.string),
                                           'inter_weight':tf.FixedLenFeature([],tf.string), 
                                           'masker_mask':tf.FixedLenFeature([],tf.string),
                                           'masker_weight':tf.FixedLenFeature([],tf.string), 
                                       })
    
    img = tf.decode_raw(features['image'], tf.uint8)
    label = tf.decode_raw(features['label'], tf.uint8)
    img = tf.reshape(img, data_shape)
    img = tf.cast(img, tf.float32)* (1./255)
    label = tf.reshape(label, [data_shape[0], data_shape[1]])
    label = tf.one_hot(indices=label,depth=2)   
    label = tf.cast(label, tf.float32)
    foreground_weight = tf.decode_raw(features['foreground_weight'], tf.float64)   
    foreground_weight = tf.reshape(foreground_weight, [data_shape[0], data_shape[1]])
    foreground_weight = tf.cast(foreground_weight, tf.float32) 
    
    
    
    inter_mask = tf.decode_raw(features['inter_mask'], tf.uint8)
    inter_weight = tf.decode_raw(features['inter_weight'], tf.float64)   
    inter_mask = tf.reshape(inter_mask, [data_shape[0], data_shape[1]])
    inter_weight = tf.reshape(inter_weight, [data_shape[0], data_shape[1]])
    inter_weight = tf.cast(inter_weight, tf.float32)  
    inter_mask = tf.one_hot(indices=inter_mask,depth=2)   
    inter_mask = tf.cast(inter_mask, tf.float32)
    
    masker_mask = tf.decode_raw(features['masker_mask'], tf.uint8)
    masker_weight = tf.decode_raw(features['masker_weight'], tf.float64)   
    masker_mask = tf.reshape(masker_mask, [data_shape[0], data_shape[1]])
    masker_weight = tf.reshape(masker_weight, [data_shape[0], data_shape[1]])
    masker_weight = tf.cast(masker_weight, tf.float32)  
    masker_mask = tf.one_hot(indices=masker_mask,depth=2)   
    masker_mask = tf.cast(masker_mask, tf.float32)
    
    
    return img, label, foreground_weight, inter_mask, inter_weight, masker_mask, masker_weight
    
