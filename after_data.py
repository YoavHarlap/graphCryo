from classification_functions import *
prev_pwd = os.getcwd()
os.chdir('/scratch/guysharon/Work/CryoEMSignalEnhancement')
from src.cryo_signal_enhance import *

#
#
# n_labeled, n_unlabeled = N_LABLED, N_UNLABLED
# dates_path = "/home/yoavharlap/work/dates/dates.txt"
# img_size = IMG_SIZE
# writeable_dir = WRITEABLE_DIR
#
# with open(dates_path, 'r') as file:
#     # Read the file content
#     content = file.read()
#     # Split the content into lines and get the last line
#     last_line = content.strip().split('\n')[-1]
#     # Extract the desired string from the last line
#     last_line = last_line.strip()
#     #print(last_line)
#
#
# last_line = last_line.split("#")
#
#
#
#
# import ast
# tmp = ast.literal_eval(last_line[1])
# starfile_fullpath = tmp['starfile']
# mrcs_fullpath = os.path.join(os.path.dirname(tmp['starfile']),tmp['mrcs'][0])
#
# import shutil
# import time
# # Source and destination paths
# src = starfile_fullpath
# time = time.strftime("%Y-%m-%d_%H:%M:%S")
#
# dst = "/home/yoavharlap/work/" + last_line[0] + "/"  + time
#
# os.mkdir(dst)
#
#
# # Copy the MRCs file to the destination folder
# shutil.copy(src, dst)
#
# src = mrcs_fullpath
#
# # Copy the MRCs file to the destination folder
# shutil.copy(src, dst)
#
#
# starfile_fullpath = dst+ "/" + starfile_fullpath.split("/")[-1]
# mrcs_fullpath = dst+ "/" + mrcs_fullpath.split("/")[-1]
#
#
#
#
# import mrcfile
# mrcs = mrcfile.open(mrcs_fullpath,'r+')
# imgs = np.array(mrcs.data)
#
# #%% add fake images
# #
# # manipulate images, for example 'imgs[0] = np.zeros((189,189))'
# #
#
# p = 2/3
# random_labels = bernoulli.rvs(p, size=n_labeled+ n_unlabeled)
# #a = np.zeros(n_labeled+ n_unlabeled-51)
# #b = np.ones(51)
# #random_labels = np.concatenate((b, a), axis=0)
#
# dataset_name = 'zero_mnist'
# dataset_path = "/data/yoavharlap/" + dataset_name
# filenames = os.listdir(dataset_path)
# #random.shuffle(filenames)
# #print('dataset path:', dataset_path)
# j = 0
# mrc = mrcfile.open("/scratch/amitayeldar/dataForTamir/Relion10028Contaminated/particles.mrcs",'r')
# outliers_imgs = np.array(mrc.data)
# mrc.close()
#
# for i in range(n_labeled+ n_unlabeled):
#        if (random_labels[i] ==0 ):
#            random_labels[i] = -1 #for classification
#            #filename = filenames[j]
#            #print(j, ":", filename)
#            #category = filename.split('.')[0]
#            #file_path = dataset_path + '/{}'.format(filename)
#            #img = imread(file_path)
#            #imgs[i] = img
#            #j = j + 1
#
#            #img = imread("/home/yoavharlap/work/outlire3.png")
#            #img = img[:, :, 0]
#            #imgs[i] = noisy(img)
#
#            imgs[i] = np.ones((img_size,img_size))
#
#            #imgs[i] = outliers_imgs[i]
# """
# imgs[200] = np.ones((img_size,img_size))
# imgs[300] = np.zeros((img_size,img_size))
# imgs[100] = outliers_imgs[0]
# imgs[101] = outliers_imgs[0]
# imgs[102] = outliers_imgs[0]
# """
#
# mrcs.set_data(imgs)
# mrcs.close()
# #%% imports for signal enhancement
# prev_pwd = os.getcwd()
# os.chdir('/scratch/guysharon/Work/CryoEMSignalEnhancement')
# from src.cryo_signal_enhance import *
# os.chdir(prev_pwd)
# import matplotlib.pyplot as plt
# import pickle
#
# #%% options for signal enhancement
# opts = {
#     #   GENERAL
#         "N":                    2147483647,
#         'verbose':              True,
#
#
#     #   PREPROCESSING
#         "downsample":           89,
#         "downsample":           img_size,
#         "batch_size":           2**15,
#         "num_coeffs":           1500,
#
#     #   EM
#         "ctf":                  True,
#         "iter":                 7,
#         "norm":                 True,
#         "em_num_inputs":        150,
#         "take_last_iter":       True,
#         "em_par":               True,
#         "em_gpu":               False,
#         "num_class_avg":        1500,
#         "num_class_avg":        1,
#
#     #   DEBUG
#         'debug_verbose':        True,
#         'random_seed':          -1,
#         'sample':               False,
#     }
#
# opts['preprocess_flags'] = [1,1,0,0,0]
#
# warnings.filterwarnings(action="ignore", message="unclosed", category=ResourceWarning)
# warnings.filterwarnings("ignore") # I get an annoying socket warning
# PrepareOpts(starfile_fullpath, writeable_dir, opts)
# preprocessed_source = PreprocessData(starfile_fullpath,opts)
#
# basis = FSPCA(preprocessed_source,opts)
#
# #%% image clustering
# opts["class_theta_step"]    = 5
# opts["num_classes"]         = 3000
# opts["num_classes"]         = 1
# opts["class_size"]          = N_LABLED+N_UNLABLED
# opts["skip_class_sort"]     = False
# opts['debug_verbose']       = False
#
# opts["class_gpu"]           = 0
# classes, classes_ref, chosen_classes, original_chosen_classes, corrs = Classify_2D(basis, opts)
#
# #%%
# #print(f"the correlation between the image {classes[0,4]} and {classes[1,4]} is {corrs[4,0,1]}")
# #print(f"the correlation between the image {classes[3,15]} and {classes[4,15]} is {corrs[15,3,4]}")
# #print("the correlation between the image 'classes[i,k]' and 'classes[j,k] is 'corrs[k,i,j]'")
#
# from matplotlib import pyplot as plt
# data = corrs[0][0:n_labeled+n_unlabeled,0:n_labeled+n_unlabeled]
# for i in range(len(data)):
#     data[i,i] = 0
# #plt.imshow(data)
# #plt.title("correlations between first 50 members of the first class (diagonal removed)")
# #plt.show()
#
# #%%
# adj_matrix = np.zeros((N_LABLED+N_UNLABLED,N_LABLED+N_UNLABLED))
# for i in range(len(classes[:, 0])):
#     print("hi ",i)
#     for j in range(i+1,len(classes[:, 0])):
#         imgi = classes[i,0]
#         imgj = classes[j,0]
#         corr = corrs[0,i,j]
#         adj_matrix[imgi][imgj] = corr
#         adj_matrix[imgj][imgi] = corr
#
#         #print("the correlation between the image 'classes[i,0]' and 'classes[j,0] is 'corrs[0,i,j]'")
#         #print(imgi,imgj,corr)
#
# for i in range(len(adj_matrix)):
#     adj_matrix[i,i] = 0
# plt.imshow(adj_matrix)
# plt.title("correlations between first 50 members of the first class (diagonal removed)")
# plt.show()
#
# #%%
#
# n_labeled, n_unlabeled = N_LABLED, N_UNLABLED
# true_labels = random_labels
# #adj_matrix = data
# labels,list_index_of_erors = run_process(n_labeled, n_unlabeled,true_labels, adj_matrix,preprocessed_source)
# #make_for_run(true_labels,n_labeled,n_unlabeled,adj_matrix)
#
#


import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import mrcfile
import matplotlib.pyplot as plt
import glob, os

images = []
outliers = []

good_path = "/data/yoavharlap/eman_particles/good"
os.chdir(good_path)
for file in glob.glob("*.mrc"):
    print(file)
    mrc = mrcfile.open(file, 'r')
    mrc_imgs = np.array(mrc.data)
    mrc.close()
    print(mrc_imgs.shape)
    print(mrc_imgs.ndim)

    if (mrc_imgs.ndim == 2):  # was just 1 pic in mrc file
        mrc_imgs = [mrc_imgs]

    for i in range(len(mrc_imgs)):
        images.append(mrc_imgs[i])

outliers_path = "/data/yoavharlap/eman_particles/outliers"
os.chdir(outliers_path)
for file in glob.glob("*.mrc"):
    print(file)
    mrc = mrcfile.open(file, 'r')
    mrc_imgs = np.array(mrc.data)
    mrc.close()
    print(mrc_imgs.shape)
    print(mrc_imgs.ndim)

    if (mrc_imgs.ndim == 2):  # was just 1 pic in mrc file
        mrc_imgs = [mrc_imgs]

    for i in range(len(mrc_imgs)):
        outliers.append(mrc_imgs[i])
import random

random.shuffle(images)
random.shuffle(outliers)

print(len(images))
print(len(outliers))

good_images = images


n_labeled, n_unlabeled = N_LABLED, N_UNLABLED
dates_path = "/home/yoavharlap/work/dates/dates.txt"
img_size = IMG_SIZE
writeable_dir = WRITEABLE_DIR

with open(dates_path, 'r') as file:
    # Read the file content
    content = file.read()
    # Split the content into lines and get the last line
    last_line = content.strip().split('\n')[-1]
    # Extract the desired string from the last line
    last_line = last_line.strip()
    # print(last_line)

last_line = last_line.split("#")

import ast

tmp = ast.literal_eval(last_line[1])
starfile_fullpath = tmp['starfile']
mrcs_fullpath = os.path.join(os.path.dirname(tmp['starfile']), tmp['mrcs'][0])

import shutil
import time

# Source and destination paths
src = starfile_fullpath
time = time.strftime("%Y-%m-%d_%H:%M:%S")

dst = "/home/yoavharlap/work/" + last_line[0] + "/" + time

os.mkdir(dst)

# Copy the MRCs file to the destination folder
shutil.copy(src, dst)

src = mrcs_fullpath

# Copy the MRCs file to the destination folder
shutil.copy(src, dst)

starfile_fullpath = dst + "/" + starfile_fullpath.split("/")[-1]
mrcs_fullpath = dst + "/" + mrcs_fullpath.split("/")[-1]

import mrcfile

mrcs = mrcfile.open(mrcs_fullpath, 'r+')
imgs = np.array(mrcs.data)

# %% add fake images
#
# manipulate images, for example 'imgs[0] = np.zeros((189,189))'
#

p = 2 / 3
random_labels = bernoulli.rvs(p, size=n_labeled + n_unlabeled)
# a = np.zeros(n_labeled+ n_unlabeled-51)
# b = np.ones(51)
# random_labels = np.concatenate((b, a), axis=0)

dataset_name = 'zero_mnist'
dataset_path = "/data/yoavharlap/" + dataset_name
filenames = os.listdir(dataset_path)
# random.shuffle(filenames)
# print('dataset path:', dataset_path)
j = 0
mrc = mrcfile.open("/scratch/amitayeldar/dataForTamir/Relion10028Contaminated/particles.mrcs", 'r')
outliers_imgs = np.array(mrc.data)
mrc.close()

imgs = []
p = 2 / 3
random_labels = bernoulli.rvs(p, size=n_labeled + n_unlabeled)

k = 0
p = 0
for i in range(n_labeled + n_unlabeled):
    if (random_labels[i] == 0):
        imgs.append(outliers[k])
        k = k + 1
    else:
        imgs.append(good_images[p])
        p = p + 1

imgs = np.array(imgs)
labels = random_labels

mrcs.set_data(imgs)
mrcs.close()
# %% imports for signal enhancement
prev_pwd = os.getcwd()
os.chdir('/scratch/guysharon/Work/CryoEMSignalEnhancement')
from src.cryo_signal_enhance import *

os.chdir(prev_pwd)
import matplotlib.pyplot as plt
import pickle

# %% options for signal enhancement
opts = {
    #   GENERAL
    "N": 2147483647,
    'verbose': True,

    #   PREPROCESSING
    "downsample": 89,
    "downsample": img_size,
    "batch_size": 2 ** 15,
    "num_coeffs": 1500,

    #   EM
    "ctf": True,
    "iter": 7,
    "norm": True,
    "em_num_inputs": 150,
    "take_last_iter": True,
    "em_par": True,
    "em_gpu": False,
    "num_class_avg": 1500,
    "num_class_avg": 1,

    #   DEBUG
    'debug_verbose': True,
    'random_seed': -1,
    'sample': False,
}

opts['preprocess_flags'] = [1, 1, 0, 0, 0]

warnings.filterwarnings(action="ignore", message="unclosed", category=ResourceWarning)
warnings.filterwarnings("ignore")  # I get an annoying socket warning
PrepareOpts(starfile_fullpath, writeable_dir, opts)
preprocessed_source = PreprocessData(starfile_fullpath, opts)

basis = FSPCA(preprocessed_source, opts)

# %% image clustering
opts["class_theta_step"] = 5
opts["num_classes"] = 3000
opts["num_classes"] = 1
opts["class_size"] = N_LABLED + N_UNLABLED
opts["skip_class_sort"] = False
opts['debug_verbose'] = False

opts["class_gpu"] = 0
classes, classes_ref, chosen_classes, original_chosen_classes, corrs = Classify_2D(basis, opts)

# %%
# print(f"the correlation between the image {classes[0,4]} and {classes[1,4]} is {corrs[4,0,1]}")
# print(f"the correlation between the image {classes[3,15]} and {classes[4,15]} is {corrs[15,3,4]}")
# print("the correlation between the image 'classes[i,k]' and 'classes[j,k] is 'corrs[k,i,j]'")

from matplotlib import pyplot as plt

data = corrs[0][0:n_labeled + n_unlabeled, 0:n_labeled + n_unlabeled]
for i in range(len(data)):
    data[i, i] = 0
# plt.imshow(data)
# plt.title("correlations between first 50 members of the first class (diagonal removed)")
# plt.show()

# %%
adj_matrix = np.zeros((N_LABLED + N_UNLABLED, N_LABLED + N_UNLABLED))
for i in range(len(classes[:, 0])):
    print("hi ", i)
    for j in range(i + 1, len(classes[:, 0])):
        imgi = classes[i, 0]
        imgj = classes[j, 0]
        corr = corrs[0, i, j]
        adj_matrix[imgi][imgj] = corr
        adj_matrix[imgj][imgi] = corr

        # print("the correlation between the image 'classes[i,0]' and 'classes[j,0] is 'corrs[0,i,j]'")
        # print(imgi,imgj,corr)

for i in range(len(adj_matrix)):
    adj_matrix[i, i] = 0
plt.imshow(adj_matrix)
plt.title("correlations between first 50 members of the first class (diagonal removed)")
plt.show()

# %%

n_labeled, n_unlabeled = N_LABLED, N_UNLABLED
true_labels = random_labels
# adj_matrix = data
labels, list_index_of_erors = run_process(n_labeled, n_unlabeled, true_labels, adj_matrix, preprocessed_source)
# make_for_run(true_labels,n_labeled,n_unlabeled,adj_matrix)


