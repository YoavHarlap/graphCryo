#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 12:40:28 2023

@author: yoavharlap
"""

NOISE_VAR_CRYO = 0
NOISE_VAR_OUTLIER = 0
CRYONUMIMAGES = 900
N_LABLED = 20
N_UNLABLED =20
SHOW = True
IMG_SIZE = 360
WRITEABLE_DIR = "/data/yoavharlap/saved_data"


import matplotlib.pyplot as plt
import numpy as np
from matplotlib.image import imread
from scipy.stats import bernoulli

import logging
import pyfftw
import matplotlib.pyplot as plt
import numpy as np
import cupy as cp
#
# from aspire.abinitio import CLSyncVoting
# from aspire.basis import FFBBasis2D, FFBBasis3D
# from aspire.classification import BFSReddyChatterjiAverager2D, RIRClass2D
# from aspire.denoising import DenoiserCov2D
# from aspire.noise import AnisotropicNoiseEstimator, CustomNoiseAdder
# from aspire.operators import FunctionFilter, RadialCTFFilter
# from aspire.reconstruction import MeanEstimator
# from aspire.source import Simulation, RelionSource, ArrayImageSource
# from aspire.source.simulation import randn
# from aspire.basis.fspca import FSPCABasis
# from aspire.storage import StarFile
# from aspire.utils.coor_trans import (
#     get_aligned_rotations,
#     get_rots_mse,
#     register_rotations,
# )
# from aspire.volume import Volume
# from aspire.source import RelionSource, Simulation

import os

#functions


def noisy(image):
    row,col= image.shape
    mean = 0
    var = 5e4
    var = NOISE_VAR_OUTLIER
    sigma = var**0.5
    gauss = np.random.normal(mean,sigma,(row,col))
    gauss = gauss.reshape(row,col)
    noisy = image + gauss
    return noisy


def make_threshold_0(list1):
    th_list1 = [1 if list1[v] > 0 else -1 for v in range(len(list1))]
    return th_list1

def eror_sum(list1, true_labels):
    th_list = make_threshold_0(list1)
    true_labels = make_threshold_0(true_labels)#make 0 -1
    erors = [0 if th_list[v] == true_labels[v] else 1 for v in range(len(th_list))]
    sums_erors = np.array(erors).sum(0)
    list_index_of_erors = []
    erors = [list_index_of_erors.append(v) if erors[v] == 1 else 1 for v in range(len(th_list))]
    return sums_erors,list_index_of_erors

def eror_sum1(list1, true_labels):
    th_list = make_threshold_0(list1)
    true_labels = make_threshold_0(true_labels)#make 0 -1

    erors = [0 if th_list[v] == true_labels[v] else 1 for v in range(len(th_list))]
    sums_erors = np.array(erors).sum(0)
    list_index_of_erors = []
    return sums_erors

def run_process(n_labeled, n_unlabeled,true_labels, adj_matrix,preprocessed_source):
    make_for_run(true_labels,n_labeled,n_unlabeled,adj_matrix)


    labels = np.zeros(n_unlabeled)
    labels = np.concatenate((true_labels[0:n_labeled], labels))
    start = init_nearest_labels(n_labeled, n_unlabeled, adj_matrix, labels)
    
    
    eror_sum1,list_index_of_erors = eror_sum(start, true_labels)
    print("error sum: ",  eror_sum1)
    print("list index errors",list_index_of_erors)
    
    #print("error sum: ", eror_sum(start, true_labels))
    #print("start: \n", start)
    #print("true labels: \n", true_labels)
    #print()
   # fig = plt.figure(figsize=(28, 28))
    j = 0
    arr = true_labels
    arr = start
    s=20
    for i in range(s):
        if arr[i] == 1:
            j = j + 1
            print("good ",j)
            #im = data[i]
            #fig.add_subplot(28, 28, j)
            if SHOW:
                preprocessed_source.images(i,1).show()  
      #plt.show()
    j = 0
   # fig = plt.figure(figsize=(28, 28))
    for i in range(s):
        if arr[i] == -1:
            j = j + 1
            print("bad ",j)
            plt.show()
            #im = data[i]
           # fig.add_subplot(28, 28, j)
            if SHOW:
                preprocessed_source.images(i,1).show()  
    #plt.show()
    eror_sum1,list_index_of_erors = eror_sum(start, true_labels)
    print("error sum: ",  eror_sum1)
    print("list index errors",list_index_of_erors)
   # print("labels", start)
    return start,list_index_of_erors



def make_for_run(true_labels,n_labeled,n_unlabeled,adj_matrix):
    error_arr = []
    list_unlabeled = list(range(1,n_labeled,10))
    for i in list_unlabeled:
        labels1 = true_labels.copy()
        labels1 = labels1[n_labeled-i:n_labeled]
        n_labeled_copy = i
        labels = np.zeros(n_unlabeled)
        labels = np.concatenate((labels1, labels))
        new_adj_matrix = adj_matrix[n_labeled-i:n_unlabeled+n_labeled, n_labeled-i:n_unlabeled+n_labeled]
        start = init_nearest_labels(n_labeled_copy, n_unlabeled, new_adj_matrix, labels)
        curr_error = eror_sum1(start, true_labels[n_labeled-i:n_unlabeled+n_labeled])
        print(i,"error sum: ",curr_error)
        error_arr.append(curr_error)
    plt.plot(list_unlabeled,error_arr)
    string = "last num of errors =" + str(curr_error)
    plt.xlabel("num of labeled data")
    plt.ylabel("num of errors")
    plt.title(string)
    plt.show()

    
    return error_arr




def init_nearest_labels(n_labeled, n_unlabeled, adj_matrix, labels):
    labels1 = labels.copy()
    for i in range(n_labeled, n_labeled + n_unlabeled):
        max_distance = adj_matrix[i, 0]
        for j in range(n_labeled):
            if adj_matrix[i, j] >= max_distance:
                #print(i, j)
                max_distance = adj_matrix[i, j]
                labels1[i] = labels[j]
                print(i,"and",j,max_distance)
    return labels1


def visual_error(n_labeled, n_unlabeled, adj_matrix, labels,true_labels,imgs):
    labels1 = labels.copy()
    for i in range(n_labeled, n_labeled + n_unlabeled):
        max_distance = adj_matrix[i, 0]
        for j in range(n_labeled):
            if adj_matrix[i, j] >= max_distance:
                #print(i, j)
                max_distance = adj_matrix[i, j]
                labels1[i] = labels[j]
                img = imgs[j]
                k=j
                print(i,"and",j,max_distance)
        fig, axarr = plt.subplots(1,2)
        str1 = "img: "+ str(i) +", tlabel: "+ str(true_labels[i]) + ", label: "+str(int(labels[i]))
        axarr[0].set_title(str1)
        str2 = "img: "+ str(k) +", tlabel: "+ str(true_labels[k]) + ", label: "+str(int(labels[k]))
        axarr[1].set_title(str2)
        axarr[0].imshow(imgs[i],cmap='gray')
        axarr[1].imshow(imgs[k],cmap='gray')
        title1 = "corr: "+ str(max_distance) +"#labeled: "+str(n_labeled) +"#unlabeled: "+str(n_unlabeled)

        if true_labels[i] != true_labels[k]:
            title1 = "corr: "+ str(max_distance) +", #labeled: "+str(n_labeled) +", #unlabeled: "+str(n_unlabeled)+"  --mistake"
        fig.suptitle(title1, fontsize=16)
    return labels1


def adj_matrix_analysis(adj_matrix, true_labels):
    plt.show()
    good_array = []
    outliers_arr = []
    mismatches_array = []
    for i in range(0,len(adj_matrix)):
        first_labeled = true_labels[i]
        if int(first_labeled) == 1:
            for j in range(0,i):
                second_labeled = true_labels[j]
                if adj_matrix[i][j]>0.7:
                    print(i,j,"have more 0.7 corr")
                #print(i,"and",j)
                if first_labeled == second_labeled:
                    if adj_matrix[i][j]>0.7:
                        print("good")
                    good_array.append(adj_matrix[i][j])
                    
                else:
                    if adj_matrix[i][j]>0.7:
                        print("mis")
                    mismatches_array.append(adj_matrix[i][j])
  
        else:
            for j in range(0,i):
                second_labeled = true_labels[j]
                if adj_matrix[i][j]>0.7:
                    print(i,j,"have more 0.7 corr")
                #print(i,"and",j)
                if first_labeled == second_labeled:
                    if adj_matrix[i][j]>0.7:
                        print("outlier")
                    outliers_arr.append(adj_matrix[i][j])
                    
                    
                    
                else:
                    if adj_matrix[i][j]>0.7:
                        print("mis")
                    mismatches_array.append(adj_matrix[i][j])
                    
                    
    plt.plot(np.sort(good_array), 'ro', label="good_pairs")
    plt.plot(np.sort(mismatches_array), 'go', label="mismatches_pairs")
    plt.plot(np.sort(outliers_arr), 'bo', label="outliers_pairs")
    plt.legend(loc="upper left")
    plt.ylabel('Correlation')
    plt.show()
                    
    return 0


def visual_error_2(n_labeled, n_unlabeled, adj_matrix, labels,true_labels,imgs):
    labels1 = labels.copy()
    counter = 0
    for i in range(n_labeled, n_labeled + n_unlabeled):
        k = np.argmax(adj_matrix[i][:n_labeled], axis=0)
        if true_labels[i] != true_labels[k]:
            counter = counter + 1 
            max_distance = adj_matrix[i][k]
            fig1, axarr = plt.subplots(1,2)
            str1 = "img: "+ str(i) +", tlabel: "+ str(true_labels[i]) + ", label: "+str(int(labels[i]))
            axarr[0].set_title(str1)
            str2 = "img: "+ str(k) +", tlabel: "+ str(true_labels[k]) + ", label: "+str(int(labels[k]))
            axarr[1].set_title(str2)
            axarr[0].imshow(imgs[i],cmap='gray')
            axarr[1].imshow(imgs[k],cmap='gray')
            title1 = "corr: "+ str(max_distance) +"#labeled: "+str(n_labeled) +"#unlabeled: "+str(n_unlabeled)

        
            title1 = "corr: "+ str(max_distance) +", #labeled: "+str(n_labeled) +", #unlabeled: "+str(n_unlabeled)+"  --mistake"
            fig1.suptitle(title1, fontsize=16)
        
    return counter



def adj_per_img_analysis(adj_matrix, true_labels):
    goods_for_good = []
    goods_for_outlier = []
    outliers_for_outlier = []
    outliers_for_good = []

    for i,labele in enumerate(true_labels):
        if labele == 1:
            for j,for_labele in enumerate(true_labels):
                if i==j:
                    c="stam"
                elif for_labele == 1:
                    goods_for_good.append([i,adj_matrix[i][j]])
                else:
                    outliers_for_good.append([i,adj_matrix[i][j]])            
        else:
            for j,for_labele in enumerate(true_labels):
                if i==j:
                    c="stam"
                elif for_labele == 1:
                    goods_for_outlier.append([i,adj_matrix[i][j]])
                else:
                    outliers_for_outlier.append([i,adj_matrix[i][j]])            
    
   
    data = np.array(outliers_for_good)
    x, y = data.T
    plt.plot(x,y, 'ko', label="outliers_for_good")
    data = np.array(goods_for_good)
    x, y = data.T
    plt.plot(x,y, 'bo', label="goods_for_good")
    
    #plt.legend(loc="upper left")
    #plt.ylabel('Correlaind = np.argpartitiontion')
    #plt.show()
    
    data = np.array(outliers_for_outlier)
    x, y = data.T
    plt.plot(x,y, 'yo', label="outliers_for_outlier")
    
    data = np.array(goods_for_outlier)
    x, y = data.T
    plt.plot(x,y, 'go', label="goods_for_outlier")

    plt.legend(loc="upper left")
    plt.ylabel('Correlation')
    plt.show()
    return 0
def ones_counter(arr):
    y = np.array(arr)
    return np.count_nonzero(y == 1)





    
def adj_per_img_analysis(adj_matrix, true_labels,show_range):
    
    k =  10
    
    correct = []
    nn_agree = []
    real = []
    for i,labele in enumerate(true_labels):
        line = adj_matrix[i]
        big_indexes = np.argpartition(line, -k)[-k:]
        ones_count = ones_counter(np.array(true_labels)[big_indexes])
        if labele == 1:
            real.append(10)
            correct.append(ones_count)
            if(true_labels[np.argmax(line)] ==1):
                nn_agree.append(int(10))
            else:
                nn_agree.append(int(0))
            
        else:
            real.append(0)
            correct.append(ones_count)
            if(true_labels[np.argmax(line)] !=1):
                nn_agree.append(int(0))
            else:
                nn_agree.append(int(10))
            
            
               
           
    plt.plot(correct[show_range:25+show_range], 'bs', label="100000 Nearest Neighbor")
    plt.plot(nn_agree[show_range:25+show_range], 'go', label="Nearest Neighbor")
    plt.plot(real[show_range:25+show_range], 'k+', label="Real label")
    #plt.legend(loc="upper left")
    plt.legend()

    plt.ylabel('Neighbors classified like image')
      
    plt.show()
    return 0

#unsimiliar = []
#for i in range(900):
 #  if(abs(nn_agree[i]-correct[i])>=5):
  #     unsimiliar.append(i)
       
       