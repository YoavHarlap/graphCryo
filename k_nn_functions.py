#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  8 21:07:34 2023

@author: yoavharlap
"""
from numpy import linalg as LA

import matplotlib.pyplot as plt
import numpy as np
import mrcfile
import glob, os
import random
from scipy.stats import bernoulli
from scipy import ndimage

def nearest_neighbor_10(labels, adj_matrix, num_labeled,n_labeled, n_unlabeled):
    num_samples = num_labeled + n_unlabeled

    predicted_labels = np.empty(n_unlabeled)

    for i in range(n_unlabeled):
        # Calculate the distances from the current unlabeled sample to all labeled samples
        distances = adj_matrix[i+n_labeled, :num_labeled]

        # Find the indices of the 10 nearest labeled samples
        nearest_indices = np.argsort(distances)[:10]

        print(labels[nearest_indices].astype(int))
        # Assign the label that occurs most frequently among the 10 nearest neighbors
        predicted_labels[i] =  np.median(labels[nearest_indices].astype(int))
        
    return predicted_labels


def adj_matrix_analysis(adj_matrix, true_labels):
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

def norm_img (img):
    return LA.norm(img)

def are_outliers (true_labels,indexes):
    are_all_outliers = True
    if(len(indexes)!=0):
        are_all_outliers = np.all(true_labels[indexes] == -1)
    return are_all_outliers


def remove_corr_data(adj_matrix,n_labeled,n_unlabeled,imgs):
    list_of_removed = []
    for i in range(n_labeled,n_labeled+n_unlabeled):
        for j in range(n_labeled+n_unlabeled):
            if(adj_matrix[i,j] > 0.7):
                list_of_removed.append(i)
                list_of_removed.append(j)

    list_of_removed = np.unique(list_of_removed)
    list_of_removed = list_of_removed[list_of_removed> n_labeled]
    print(list_of_removed)
    
    if(len(list_of_removed)!=0):
        new_imgs = np.delete(imgs, list_of_removed,axis=0)
    else:
        new_imgs = imgs.copy()
    return new_imgs,list_of_removed
    
                
    

def normalize_img(img, desired_norm = 50):
    real_norm = LA.norm(img, 'fro')
    new_img = img * desired_norm / real_norm
    return new_img

    
def img_rotate_avr(img):
    new1 = np.rot90(img)
    new2 = np.rot90(new1)
    new3 = np.rot90(new2)
    new = (new1+new2+new3+img)/4
    return new

def load_mrcs_from_path_to_arr(path):
    images = []
    images_filenames = []
    os.chdir(path)
    for file in glob.glob("*.mrc"):
        #print(file)
        mrc = mrcfile.open(file, 'r')
        mrc_imgs = np.array(mrc.data)
        mrc.close()
        #print(mrc_imgs.shape)
        #print(mrc_imgs.ndim)

        if (mrc_imgs.ndim == 2):  # was just 1 pic in mrc file
            mrc_imgs = [mrc_imgs]

        for i in range(len(mrc_imgs)):
            images.append(mrc_imgs[i])
            images_filenames.append(file)
    return images


def make_imgs_arr_from_labels(labels, good_imgs, outliers_imgs):
    imgs = []
    k = 0
    p = 0
    for i in range(len(labels)):
        if (labels[i] != 1):
            labels[i] = -1
            imgs.append(outliers_imgs[k])
            k = k + 1
        else:
            imgs.append(good_imgs[p])
            p = p + 1
    
    imgs = np.array(imgs)
    return imgs,labels


    # %% imports for signal enhancement

def calc_similarity_matrix(imgs):
    img_size = imgs[0].shape[0]
    prev_pwd = os.getcwd()
    os.chdir('/scratch/guysharon/Work/CryoEMSignalEnhancement')
    from src.cryo_signal_enhance import Classify_2D,CreateStarFileFromImages_WithoutCTFData,PrepareOpts,PreprocessData,FSPCA
    os.chdir(prev_pwd)
    # %%

    starfile_fullpath = '/data/yoavharlap/saved_data/starfile.star'
    writeable_dir = "/data/yoavharlap/saved_data"  # this is where data will be saved
    CreateStarFileFromImages_WithoutCTFData(starfile_fullpath, imgs)

    opts = {
        #   GENERAL
        "N": 2147483647,
        'verbose': True,

        #   PREPROCESSING
        "downsample":           89,
        "downsample": img_size,
        "batch_size": 2 ** 15,
        "num_coeffs": 1500,
        'preprocess_flags': [1, 1, 0, 0, 0],
        "num_class_avg":        1500,
        "num_class_avg":        1,

        #   DEBUG
        'debug_verbose': True,
        'random_seed': -1,
        'sample': False,
    }
    import warnings
    warnings.filterwarnings(action="ignore", message="unclosed", category=ResourceWarning)
    warnings.filterwarnings("ignore")  # I get an annoying socket warning

    PrepareOpts(starfile_fullpath, writeable_dir, opts)
    preprocessed_source = PreprocessData(starfile_fullpath, opts)
    basis = FSPCA(preprocessed_source, opts)



    # %% image clustering
    len_images = imgs.shape[0]
    opts["class_theta_step"] = 5
    opts["num_classes"] = 3000
    opts["num_classes"] = 1
    opts["class_size"] = len_images
    opts["skip_class_sort"] = False
    opts['debug_verbose'] = False

    opts["class_gpu"] = 0
    classes, classes_ref, chosen_classes, original_chosen_classes, corrs = Classify_2D(basis, opts)
    # print("the correlation between the image 'classes[i,k]' and 'classes[j,k] is 'corrs[k,i,j]'")

    adj_matrix = np.zeros((len_images, len_images))
    for i in range(len(classes[:, 0])):
        for j in range(i + 1, len(classes[:, 0])):
            imgi = classes[i, 0]
            imgj = classes[j, 0]
            corr = corrs[0, i, j]
            adj_matrix[imgi][imgj] = corr
            adj_matrix[imgj][imgi] = corr

    for i in range(len(adj_matrix)):
        adj_matrix[i, i] = 0
    #plt.imshow(adj_matrix)
    #plt.title("correlations")
    #plt.show()
    
    return adj_matrix


def nearest_row_neighbor_from_labeled(i,adj_matrix, n_labeled, n_unlabeled):
    neighbor_index = np.argmax(adj_matrix[i,:n_labeled])
    #max_value = np.max(adj_matrix[i,:n_labeled])
    return neighbor_index

def get_indexes_k_nearest_row_neighbor_from_labeled(i,k,adj_matrix, n_labeled, n_unlabeled):
    line = adj_matrix[i,:n_labeled]
    neighbor_indexes = np.argpartition(line, -k)[-k:]  
    return neighbor_indexes


def nearest_neighbor(labels, adj_matrix, n_labeled, n_unlabeled):
    # Initialize the predicted labels for unlabeled samples
    predicted_labels = np.copy(labels)
    
    # Extract the similarity matrix between labeled and unlabeled samples
    similarity_matrix = adj_matrix[:n_labeled, n_labeled:n_labeled + n_unlabeled]
    
    # Find the nearest labeled sample for each unlabeled sample
    nearest_labeled_indices = np.argmax(similarity_matrix, axis=0)
    
    # Assign labels to unlabeled samples based on their nearest labeled neighbors
    predicted_labels = labels[nearest_labeled_indices]
    
    return predicted_labels



def nearest_neighbor_from_labeled(labels,adj_matrix, n_labeled, n_unlabeled):
    labels1 = labels.copy()
    for i in range(n_labeled,n_unlabeled+n_labeled):
        neighbor_index = nearest_row_neighbor_from_labeled(i,adj_matrix, n_labeled, n_unlabeled)
        labels1[i]=labels[neighbor_index]  
    return labels1

def get_indexes_k_nearest_neighbor_from_labeled(k,adj_matrix, n_labeled, n_unlabeled):
    neighbor_indexes = []
    for i in range(n_labeled,n_unlabeled+n_labeled):
        neighbor_indexes.append(get_indexes_k_nearest_row_neighbor_from_labeled(i,k,adj_matrix, n_labeled, n_unlabeled)) 
    return neighbor_indexes

def get_labels_median_k_nn(labels,k,adj_matrix, n_labeled, n_unlabeled):
    nn_knn_not_similar = []
    neighbor_indexes = get_indexes_k_nearest_neighbor_from_labeled(k,adj_matrix, n_labeled, n_unlabeled)
    j=0
    nn_labels = np.zeros(len(labels))
    nn_labels[:n_labeled] = labels[:n_labeled]
    
    new_algo = np.zeros(len(labels))
    new_algo[:n_labeled] = labels[:n_labeled]
    for i in range(n_labeled,n_unlabeled+n_labeled):
        labels[i] = np.median(labels[neighbor_indexes[j]])
        if(labels[i] == 0):
            labels[i] = -1
        neighbor_index = nearest_row_neighbor_from_labeled(i,adj_matrix, n_labeled, n_unlabeled)
        nn_labels[i]=labels[neighbor_index]  
        j=j+1
        if(labels[i]!=nn_labels[i]):
            nn_knn_not_similar.append(i)
            if(nn_labels[i] ==1):
                new_algo[i] = nn_labels[i]
            elif(labels[i] ==1):
                new_algo[i] = labels[i]
            else:
                new_algo[i] = labels[i]
        else:
            new_algo[i] = labels[i]
    return labels,new_algo

def get_labels_corr_threshold(labels,k,adj_matrix, n_labeled, n_unlabeled,threshold = 0.5):
    j=0
    ct_labels = np.zeros(len(labels))
    ct_labels[:n_labeled] = labels[:n_labeled]
    
    for i in range(n_labeled,n_unlabeled+n_labeled):
        if(np.any(adj_matrix[i] >threshold)):
            ct_labels[i] = -1
    return ct_labels

def ones_counter(arr):
    y = np.array(arr)
    return np.count_nonzero(y == 1)


def get_labels_n_outliers_from_k_nn(labels,n_outliers,k,adj_matrix, n_labeled, n_unlabeled):
    neighbor_indexes = get_indexes_k_nearest_neighbor_from_labeled(k,adj_matrix, n_labeled, n_unlabeled)
    j=0
    new_labels = np.zeros(len(labels))
    new_labels[:n_labeled] = labels[:n_labeled]
    for i in range(n_labeled,n_labeled+n_unlabeled):
        n_goods_neighbors = ones_counter(labels[neighbor_indexes[j]])
        n_outliers_neighbors = k-n_goods_neighbors
        j = j+1
        if(n_outliers_neighbors>=n_outliers):
            new_labels[i] = -1
        else:
            new_labels[i] = 1
            
    return new_labels
            
    
def visual_error_2(n_labeled, n_unlabeled, adj_matrix, labels,true_labels,imgs):
    labels1 = labels.copy()
    counter = 0
    for i in range(n_labeled, n_labeled + n_unlabeled):
        k = np.argmax(adj_matrix[i,:n_labeled], axis=0)
        if labels[i] != true_labels[i]:
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
            plt.show()
    return counter

def run_process(n_labeled, n_unlabeled,true_labels, adj_matrix,preprocessed_source):
    #make_for_run(true_labels,n_labeled,n_unlabeled,adj_matrix)


    labels = np.zeros(n_unlabeled)
    labels = np.concatenate((true_labels[0:n_labeled], labels))
    start = init_nearest_labels(n_labeled, n_unlabeled, adj_matrix, labels)
    
    
    eror_sum1,list_index_of_erors = eror_sum(start, true_labels)
    print("error sum: ",  eror_sum1)
    print("list index errors",list_index_of_erors)
    
    
    eror_sum1,list_index_of_erors = eror_sum(start, true_labels)
    print("error sum: ",  eror_sum1)
    print("list index errors",list_index_of_erors)
   # print("labels", start)
    return start,list_index_of_erors


def make_threshold_0(list1):
    th_list1 = [1 if list1[v] > 0 else -1 for v in range(len(list1))]
    return th_list1


def init_nearest_labels(n_labeled, n_unlabeled, adj_matrix, labels):
    labels1 = labels.copy()
    for i in range(n_labeled, n_labeled + n_unlabeled):
        max_distance = adj_matrix[i, 0]
        for j in range(n_labeled):
            if adj_matrix[i, j] >= max_distance:
                #print(i, j)
                max_distance = adj_matrix[i, j]
                labels1[i] = labels[j]
                #print(i,"and",j,max_distance)
    return labels1



def eror_sum(list1, true_labels):
    th_list = make_threshold_0(list1)
    true_labels = make_threshold_0(true_labels)#make 0 -1
    erors = [0 if th_list[v] == true_labels[v] else 1 for v in range(len(th_list))]
    sums_erors = np.array(erors).sum(0)
    list_index_of_erors = []
    erors = [list_index_of_erors.append(v) if erors[v] == 1 else 1 for v in range(len(th_list))]
    return sums_erors,list_index_of_erors

def is_square(m):
    return m.shape[0] == m.shape[1]

    
def calc_sum_corr(adj_matrix1,true_labels1):
	outliers_sums_arr = []
	good_sums_arr = []
	for i,curr_label in enumerate (true_labels1):
		if curr_label == 1:
			good_sums_arr.append(np.sum(adj_matrix1[i]))
		else:
			outliers_sums_arr.append(np.sum(adj_matrix1[i]))

	plt.plot(outliers_sums_arr,'o', label="outliers_sums_arr")
	plt.plot(good_sums_arr,'o', label="good_sums_arr")
	plt.legend()

	plt.xlabel("images")
	plt.ylabel("corr")
	title = "sum corr with image" 
	plt.title(title)
	plt.show()

import imagehash
from PIL import Image
import numpy as np
from PIL import Image
from matplotlib import cm

def calc_hash(imgs, true_labels1):
    outliers_sums_arr = []
    good_sums_arr = []
    for i, curr_label in enumerate(true_labels1):
        imi = Image.fromarray(np.uint8(cm.gist_earth(imgs[i]) * 255))
        hashi = imagehash.average_hash(imi)
        for j, iter_label in enumerate(true_labels1[i + 1:]):

            imj = Image.fromarray(np.uint8(cm.gist_earth(imgs[j]) * 255))

            hashj = imagehash.average_hash(imj)
            if curr_label == iter_label:

                good_sums_arr.append(abs(hashj - hashi))
            else:
                outliers_sums_arr.append(abs(hashj - hashi))
    plt.plot(outliers_sums_arr,'o', label="not same class")
    plt.plot(good_sums_arr,'o', label="same class")
    plt.legend()
   
    plt.xlabel("images")
    plt.ylabel("different")
    title = " " 
    plt.title(title)
    plt.show()    
    

def filters(imgs,true_labels):
    
    for i in range(len(imgs)):
        org = img_rotate_avr(imgs[i])
        blur1 = ndimage.median_filter(org,100)
        blur2 = ndimage.rank_filter(org,70,70)
    
    
        img = imgs[i]
        ########
        # Get x-gradient in "sx"
        sx = ndimage.sobel(img, axis=0, mode='constant')
        # Get y-gradient in "sy"
        sy = ndimage.sobel(img, axis=1, mode='constant')
        # Get square root of sum of squares
        sobel1 = np.hypot(sx, sy)
        sobel1 = sobel1 * 100
        # print(sobel1)
        # Hopefully see some edges
        # plt.imshow(sobel1, cmap=plt.cm.gray)
        # plt.show()
    
        ##########
        # Define kernel for x differences
        kx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        # Define kernel for y differences
        ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        # Perform x convolution
        x = ndimage.convolve(img, kx)
        # Perform y convolution
        y = ndimage.convolve(img, ky)
        sobel = np.hypot(x, y)
        # plt.imshow(sobel, cmap=plt.cm.gray)
        # plt.show()
        if true_labels[i] == 1:
            label = "'good'"
        else:
            label = "'outlier'"
        f_size = 12
        fig, ax = plt.subplots(1,6,figsize=(10,10))
        ax[0].imshow(imgs[i], cmap='gray')
        ax[0].set_title(label+' gauss' , fontsize=f_size)
        ax[1].imshow(org, cmap='gray')
        ax[1].set_title('rot and avr ', fontsize = f_size)
        ax[2].imshow(blur1, cmap = 'gray')
        ax[2].set_title('median filter', fontsize = f_size);
        ax[3].imshow(blur2, cmap='gray')
        ax[3].set_title('rank filter', fontsize=f_size);
        ax[4].imshow(sobel1, cmap='gray')
        ax[4].set_title('gradient', fontsize=f_size);
        ax[5].imshow(sobel, cmap='gray')
        ax[5].set_title('gradient2', fontsize=f_size);
        plt.show()  

    
    
    
    
    
     