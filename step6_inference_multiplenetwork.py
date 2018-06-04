# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 09:29:58 2018

@author: Xie Lipeng
"""

import numpy as np
from utils.write_read_tfrecord import *
import tensorflow as tf
import tensorlayer as tl 
from PIL import Image
import matplotlib.pyplot as plt
import glob
import scipy.misc
import os
import scipy.io as sio 
import time
from nets.construct_model_multiple_networks import *
from scipy import ndimage as ndi
from skimage.morphology import watershed
from skimage.morphology import remove_small_objects
from skimage.morphology import label
from skimage.color import label2rgb
from skimage.feature import peak_local_max
from skimage import color, io, filters as filters
from utils.Postprocessing import Watershed_Dynamic
import skimage.morphology as morphology
from skimage.morphology import binary_erosion as erosion


def Watershed_Proposed(mask,maskers):
     maskers = maskers & mask
     maskers = label(maskers)
     distance = ndi.distance_transform_edt(mask)
     labels = watershed(-distance, markers=maskers, mask=mask)
     labels = remove_small_objects(labels, 100)
     labelsrgb = label2rgb(labels,bg_label = 0, bg_color=(0.2, 0.5, 0.6))
     return labels, labelsrgb

def Watershed_Classical(mask):
    distance = ndi.distance_transform_edt(mask)
    local_maxi = peak_local_max(distance, indices=False,footprint=np.ones((2, 2)), 
                            labels=mask)
    markers = ndi.label(local_maxi)[0]
    labels = watershed(-distance, markers, mask=mask)
    labels = remove_small_objects(labels, 100)
    labelsrgb = label2rgb(labels,bg_label = 0, bg_color=(0.2, 0.5, 0.6))
    return labels, labelsrgb

def condition_erosion(mask,erosion_structure,threshold):
    mask_process = np.zeros(np.shape(mask))
    mask_label,N = label(mask,return_num=True)
    for i in range(1,N+1):
        mask_temp = (mask_label==i)
        while np.sum(mask_temp)>=threshold:
            mask_temp = erosion(mask_temp,erosion_structure)
        mask_process = mask_process + mask_temp
    return mask_process
        
def Watershed_Condition_erosion(mask):
    fine_structure = morphology.diamond(1)
    coarse_structure = morphology.diamond(3)
    coarse_structure[3,0]=0
    coarse_structure[3,6]=0

    #==========step1 coarse erosion=============
    seed_mask = condition_erosion(mask,coarse_structure,200)
    #==========step2 fine erosion=============
    seed_mask = condition_erosion(seed_mask,fine_structure,50)
    
    distance = ndi.distance_transform_edt(mask)
    markers = label(seed_mask)
    labels = watershed(-distance, markers, mask=mask)
    labelsrgb = label2rgb(labels,bg_label = 0, bg_color=(0.2, 0.5, 0.6))
#    markersrgb = label2rgb(markers,bg_label = 0, bg_color=(0.2, 0.5, 0.6))
    return labels, labelsrgb    

    
    
   
    
    
    
    
    
    
#=========================step1: Set parameters===============================
DatasetName = "BNS" # "MICCAI2017" or "BNS"
NetworkList = ['FCN8s','HED','Unet','SharpMask','DIMAN'] #'FCN8s','HED','Unet','SharpMask','DIMAN'
model_file = 'model_parameters.ckpt'
data_shape = [512,512,3] 
is_train = False
reuse= False
c = [1,1,1]
datapath = './data/'+ DatasetName +'/'  #the path to store patches and lmdb
mean_file_name = datapath +'train_1_1.tfrecords_mean.npy'


if DatasetName=='MICCAI2017':
    datapath_images = '../Nuclei_segmentation_testing/'
    subfoldname_images = ['gbm','hnsc','lgg','lung']
else:
    datapath_images = '../BNS_Nuclei_Data/'
    subfoldname_images = os.listdir(datapath_images)

    
batch_size = 1
x = tf.placeholder(tf.float32, shape=[batch_size, data_shape[0],data_shape[1], 3])   # [batch_size, height, width, channels]
y_ = tf.placeholder(tf.float32, shape=[batch_size,data_shape[0],data_shape[1], 2])
fw = tf.placeholder(tf.float32, shape=[batch_size,data_shape[0],data_shape[1]])
im = tf.placeholder(tf.float32, shape=[batch_size, data_shape[0],data_shape[1], 2])   # [batch_size, height, width, channels]
iw = tf.placeholder(tf.float32, shape=[batch_size,data_shape[0],data_shape[1]])
mm = tf.placeholder(tf.float32, shape=[batch_size, data_shape[0],data_shape[1], 2])   # [batch_size, height, width, channels]    
mw = tf.placeholder(tf.float32, shape=[batch_size,data_shape[0],data_shape[1]])    
    
   
#=========================step3: Creat network===============================
# Define the batchsize at the begin, you can give the batchsize in x and y_
for NetworkName in NetworkList:
    model_path = './checkpoints/'+ DatasetName+ '/' + NetworkName +'/'+ model_file
    network_scopename = NetworkName
    save_path = './outputs/'+ DatasetName+ '/'+ NetworkName+ '/'
    totaltime = 0
    totaltime_classicalwatershed=0
    totaltime_erosionwatershed=0
    totaltime_dynamicwatershed=0
    totaltime_proposedwatershed=0 
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    
    if NetworkName == 'DIMAN' :
        network,op_class_fg,op_prob_fg,op_class_interval,op_class_masker,op_class_refine,op_prob_refine = eval('model_VGG16_' + NetworkName + '(x,y_,fw,im,iw,mm,mw,c,batch_size,data_shape,reuse=reuse,mean_file_name=mean_file_name, is_train = is_train,network_scopename=network_scopename)')
        sess = tf.Session()
        init = tf.initialize_all_variables()  
        sess.run(init)
        print("Load existing model: " + "!"*10)
        saver = tf.train.Saver(network.all_params)
        saver.restore(sess, model_path)
        Num_img = 0
        #=========================step4: Load image===============================
        for subfoldname in subfoldname_images:
            for imgmaskname in sorted(glob.glob("%s*_mask.png" %(datapath_images+subfoldname+'/testing-set/'))):
                imgpathname = imgmaskname[:-9]
                if DatasetName=='MICCAI2017':
                    imgname = imgmaskname[-16:-9]
                else: 
                    imgname = imgpathname.split('\\')
                    imgname = imgname[1].split('_')
                    imgname = imgname[1]                 
                image = Image.open(imgpathname+'.png')
                img_shape = image.size
                if img_shape[0]!=data_shape[0] or img_shape[1]!=data_shape[1]:
                    Flag_imgreshape = True
                    image = image.resize((data_shape[0],data_shape[1]))
                else:
                    Flag_imgreshape = False
                image = np.array(image)/255
                image = image[:,:,0:3]
                image = image[np.newaxis,:]
                feed_dict = {x: image}
                #======time=======================
                start_time = time.time()
                prediction_class_fg_out,prediction_prob_fg_out,prediction_class_interval_out,prediction_class_masker_out, prediction_class_refine_out,prediction_prob_refine_out = sess.run(
                                [op_class_fg,op_prob_fg,op_class_interval,op_class_masker,op_class_refine,op_prob_refine], feed_dict=feed_dict) 
                totaltime = totaltime + (time.time() - start_time)
                
                prediction_prob_refine = prediction_prob_refine_out[0,:,:,0]
                prediction_class_refine = prediction_class_refine_out[0,:,:,0]
                prediction_class_interval = prediction_class_interval_out[0,:,:,0]
                prediction_class_masker = prediction_class_masker_out[0,:,:,0]
                prediction_class_fg = prediction_class_fg_out[0,:,:,0]
                prediction_prob_fg = prediction_prob_fg_out[0,:,:,0]        

                Num_img = Num_img + 1
                if Flag_imgreshape:
                    prediction_class_refine = Image.fromarray(np.uint8(prediction_class_refine))
                    prediction_class_refine = prediction_class_refine.resize(img_shape)
                    prediction_class_refine = np.int64(np.array(prediction_class_refine))                   
                    
                    prediction_class_interval = Image.fromarray(np.uint8(prediction_class_interval))
                    prediction_class_interval = prediction_class_interval.resize(img_shape)
                    prediction_class_interval = np.int64(np.array(prediction_class_interval))                
                    
                    prediction_class_masker = Image.fromarray(np.uint8(prediction_class_masker))
                    prediction_class_masker = prediction_class_masker.resize(img_shape)
                    prediction_class_masker = np.int64(np.array(prediction_class_masker))
                    
                    prediction_class_fg = Image.fromarray(np.uint8(prediction_class_fg))
                    prediction_class_fg = prediction_class_fg.resize(img_shape)
                    prediction_class_fg = np.int64(np.array(prediction_class_fg))
                    
                    prediction_prob_fg = Image.fromarray(np.float32(prediction_prob_fg))
                    prediction_prob_fg = prediction_prob_fg.resize(img_shape)
                    prediction_prob_fg = np.float32(np.array(prediction_prob_fg))
                    
                    prediction_prob_refine = Image.fromarray(np.float32(prediction_prob_refine))
                    prediction_prob_refine = prediction_prob_refine.resize(img_shape)
                    prediction_prob_refine = np.float32(np.array(prediction_prob_refine))
                    
                #================fg===================
                fg_labels = label(prediction_class_fg)
                fg_labelsrgb = label2rgb(fg_labels,bg_label = 0, bg_color=(0.2, 0.5, 0.6))
                scipy.misc.imsave(save_path + subfoldname +'_'+imgname +'_fgrgb_result.png',fg_labelsrgb)
                sio.savemat(save_path + subfoldname +'_'+imgname +'_fg_result.mat', {'fg_result': fg_labels})
                #================fg+classical watershed===================
                labels, labelsrgb = Watershed_Classical(prediction_class_fg)
                scipy.misc.imsave(save_path + subfoldname +'_'+imgname +'_fg_watershed_result.png',labels>0)
                scipy.misc.imsave(save_path + subfoldname +'_'+imgname +'_fg_watershedrgb_result.png',labelsrgb)
                sio.savemat(save_path + subfoldname +'_'+imgname +'_fg_watershed_result.mat', {'fg_watershed_result': labels}) 
                #================fg+dynamic watershed===================
                labels, labelsrgb = Watershed_Dynamic(prediction_prob_fg)
                scipy.misc.imsave(save_path + subfoldname +'_'+imgname +'_fg_dynamicwatershed_result.png',labels>0)
                scipy.misc.imsave(save_path + subfoldname +'_'+imgname +'_fg_dynamicwatershedrgb_result.png',labelsrgb)
                sio.savemat(save_path + subfoldname +'_'+imgname +'_fg_dynamicwatershed_result.mat', {'fg_dynamicwatershed_result': labels})
                #================fg+Condition_erosion watershed===================
                labels, labelsrgb = Watershed_Condition_erosion(prediction_class_fg)
                scipy.misc.imsave(save_path + subfoldname +'_'+imgname +'_fg_erosionwatershed_result.png',labels>0)
                scipy.misc.imsave(save_path + subfoldname +'_'+imgname +'_fg_erosionwatershedrgb_result.png',labelsrgb)
                sio.savemat(save_path + subfoldname +'_'+imgname +'_fg_erosionwatershed_result.mat', {'fg_erosionwatershed_result': labels})    
                #================fg+interval===================
                refine_labels = label(prediction_class_refine)
                refine_labelsrgb = label2rgb(refine_labels,bg_label = 0, bg_color=(0.2, 0.5, 0.6))
                scipy.misc.imsave(save_path + subfoldname +'_'+imgname +'_refinergb_result.png',refine_labelsrgb)
                sio.savemat(save_path + subfoldname +'_'+imgname +'_refine_result.mat', {'refine_result': refine_labels})
                #================fg+interval+classical watershed===================
                start_time = time.time()
                labels, labelsrgb = Watershed_Classical(prediction_class_refine)
                scipy.misc.imsave(save_path + subfoldname +'_'+imgname +'_refine_watershed_result.png',labels>0)
                scipy.misc.imsave(save_path + subfoldname +'_'+imgname +'_refine_watershedrgb_result.png',labelsrgb)
                sio.savemat(save_path + subfoldname +'_'+imgname +'_refine_watershed_result.mat', {'refine_watershed_result': labels})
                totaltime_classicalwatershed = totaltime + (time.time() - start_time)
                #================fg+interval+dynamic watershed===================
                start_time = time.time()
                labels, labelsrgb = Watershed_Dynamic(prediction_prob_refine)
                scipy.misc.imsave(save_path + subfoldname +'_'+imgname +'_refine_dynamicwatershed_result.png',labels>0)
                scipy.misc.imsave(save_path + subfoldname +'_'+imgname +'_refine_dynamicwatershedrgb_result.png',labelsrgb)
                sio.savemat(save_path + subfoldname +'_'+imgname +'_refine_dynamicwatershed_result.mat', {'refine_dynamicwatershed_result': labels})
                totaltime_dynamicwatershed = totaltime + (time.time() - start_time)
                #================fg+interval+Condition_erosion watershed===================
                start_time = time.time()
                labels, labelsrgb = Watershed_Condition_erosion(prediction_class_refine)
                scipy.misc.imsave(save_path + subfoldname +'_'+imgname +'_refine_erosionwatershed_result.png',labels>0)
                scipy.misc.imsave(save_path + subfoldname +'_'+imgname +'_refine_erosionwatershedrgb_result.png',labelsrgb)
                sio.savemat(save_path + subfoldname +'_'+imgname +'_refine_erosionwatershed_result.mat', {'refine_erosionwatershed_result': labels})    
                totaltime_erosionwatershed = totaltime + (time.time() - start_time)
                #================fg+interval+marker+proposed watershed===================
                start_time = time.time()
                labels, labelsrgb = Watershed_Proposed(prediction_class_refine, prediction_class_masker)
                scipy.misc.imsave(save_path + subfoldname +'_'+imgname +'_fg_interval_marker_watershed_result.png',labels>0)
                scipy.misc.imsave(save_path + subfoldname +'_'+imgname +'_fg_interval_marker_watershedrgb_result.png',labelsrgb)
                sio.savemat(save_path + subfoldname +'_'+imgname +'_fg_interval_marker_watershed_result.mat', {'fg_interval_marker_watershed_result': labels})  
                totaltime_proposedwatershed = totaltime + (time.time() - start_time)
                    
                scipy.misc.imsave(save_path + subfoldname +'_'+imgname +'_masker_result.png',prediction_class_masker)
                scipy.misc.imsave(save_path + subfoldname +'_'+imgname +'_refine_result.png',prediction_class_refine)
                scipy.misc.imsave(save_path + subfoldname +'_'+imgname +'_interval_result.png',prediction_class_interval)
                scipy.misc.imsave(save_path + subfoldname +'_'+imgname +'_fg_result.png',prediction_class_fg)
                    
        #=========================step4: Train network===============================
        sess.close()
        print(" Mean time Network: %f" % (totaltime/Num_img))
        print(" Mean time with classical watershed: %f" % (totaltime_classicalwatershed/Num_img))
        print(" Mean time with erosion watershed: %f" % (totaltime_erosionwatershed/Num_img))
        print(" Mean time with dynamic watershed: %f" % (totaltime_dynamicwatershed/Num_img))
        print(" Mean time with proposed watershed: %f" % (totaltime_proposedwatershed/Num_img))
        
    else:    
        network,op_class_fg,op_prob_fg = eval('model_VGG16_'+NetworkName+'(x,y_,fw,batch_size,data_shape,reuse=reuse,mean_file_name=mean_file_name, is_train = is_train,network_scopename=network_scopename)')
        sess = tf.Session()
        init = tf.initialize_all_variables()  
        sess.run(init)
        print("Load existing model:" + "!"*10)
        saver = tf.train.Saver(network.all_params) 
        saver.restore(sess, model_path)
        Num_img = 0
        start_time = time.time()
        totaltime = 0
        totaltime_watershed = 0
        #=========================step4: Load image===============================
        for subfoldname in subfoldname_images:
            for imgmaskname in sorted(glob.glob("%s*_mask.png" %(datapath_images+subfoldname+'/testing-set/'))):
                imgpathname = imgmaskname[:-9]
                if DatasetName=='MICCAI2017':
                    imgname = imgmaskname[-16:-9]
                else: 
                    imgname = imgpathname.split('\\')
                    imgname = imgname[1].split('_')
                    imgname = imgname[1]  
                image = Image.open(imgpathname+'.png')
                img_shape = image.size
                if img_shape[0]!=data_shape[0] or img_shape[1]!=data_shape[1]:
                    Flag_imgreshape = True
                    image = image.resize((data_shape[0],data_shape[1]))
                else:
                    Flag_imgreshape = False
                image = np.array(image)/255
                image = image[:,:,0:3]
                image = image[np.newaxis,:]
                feed_dict = {x: image}

                #======time=======================
                start_time = time.time()
                prediction_class_fg_out,prediction_prob_fg_out = sess.run([op_class_fg,op_prob_fg], feed_dict=feed_dict)
                totaltime = totaltime + (time.time() - start_time)
                prediction_class_fg = prediction_class_fg_out[0,:,:,0]
                prediction_prob_fg =  prediction_prob_fg_out[0,:,:,0]
                Num_img = Num_img + 1
                if Flag_imgreshape:
                    prediction_class_fg = Image.fromarray(np.float32(prediction_class_fg))
                    prediction_class_fg = prediction_class_fg.resize(img_shape)
                    prediction_class_fg = np.float32(np.array(prediction_class_fg))
                    
                    prediction_prob_fg = Image.fromarray(np.float32(prediction_prob_fg))
                    prediction_prob_fg = prediction_prob_fg.resize(img_shape)
                    prediction_prob_fg = np.float32(np.array(prediction_prob_fg))
                    
                scipy.misc.imsave(save_path + subfoldname +'_'+imgname +'_fg_result.png',prediction_class_fg)
                #================fg===================
                fg_labels = label(prediction_class_fg)
                fg_labelsrgb = label2rgb(fg_labels,bg_label = 0, bg_color=(0.2, 0.5, 0.6))
                scipy.misc.imsave(save_path + subfoldname +'_'+imgname +'_fgrgb_result.png',fg_labelsrgb)
                sio.savemat(save_path + subfoldname +'_'+imgname +'_fg_result.mat', {'fg_result': fg_labels})
                #================fg+classical watershed===================
                start_time = time.time()
                labels, labelsrgb = Watershed_Classical(prediction_class_fg)
                scipy.misc.imsave(save_path + subfoldname +'_'+imgname +'_fg_watershed_result.png',labels>0)
                scipy.misc.imsave(save_path + subfoldname +'_'+imgname +'_fg_watershedrgb_result.png',labelsrgb)
                sio.savemat(save_path + subfoldname +'_'+imgname +'_fg_watershed_result.mat', {'fg_watershed_result': labels}) 
                totaltime_watershed = totaltime_watershed + (time.time() - start_time)
                #================fg+dynamic watershed===================
                labels, labelsrgb = Watershed_Dynamic(prediction_prob_fg)
                scipy.misc.imsave(save_path + subfoldname +'_'+imgname +'_fg_dynamicwatershed_result.png',labels>0)
                scipy.misc.imsave(save_path + subfoldname +'_'+imgname +'_fg_dynamicwatershedrgb_result.png',labelsrgb)
                sio.savemat(save_path + subfoldname +'_'+imgname +'_fg_dynamicwatershed_result.mat', {'fg_dynamicwatershed_result': labels})
                #================fg+Condition erosion watershed===================
                labels, labelsrgb = Watershed_Condition_erosion(prediction_class_fg)
                scipy.misc.imsave(save_path + subfoldname +'_'+imgname +'_fg_erosionwatershed_result.png',labels>0)
                scipy.misc.imsave(save_path + subfoldname +'_'+imgname +'_fg_erosionwatershedrgb_result.png',labelsrgb)
                sio.savemat(save_path + subfoldname +'_'+imgname +'_fg_erosionwatershed_result.mat', {'fg_erosionwatershed_result': labels})
               
        #=========================step4: Train network===============================
        sess.close()
        print(" Mean time Network: %f" % (totaltime/Num_img))
        print(" Mean time with watershed: %f" % (totaltime_watershed/Num_img))
            





  