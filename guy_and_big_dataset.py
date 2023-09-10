import matplotlib.pyplot as plt
import numpy as np
from classification_functions import run_process,adj_matrix_analysis,visual_error_2,eror_sum1


IMG_SIZE = 256
N_LABLED = 60
N_UNLABLED = 30

# Paths to your numpy files
outliers_file_path = "/data/yoavharlap/10028_classification/outliers_images.npy"
particles_file_path = "/data/yoavharlap/10028_classification/particles_images.npy"

outliers_data = np.load(outliers_file_path)
particles_data = np.load(particles_file_path)
data = np.concatenate((outliers_data, particles_data), axis=0)
labels = np.concatenate((np.ones(len(outliers_data)), np.zeros(len(particles_data))))
train_ratio = 0.9
total_samples = len(labels)
train_samples = int(train_ratio * total_samples)

# Create an index array to shuffle data and labels in the same way
shuffle_indices = np.arange(len(data))
np.random.shuffle(shuffle_indices)

# Shuffle data and labels using the shuffled indices
shuffled_data = data[shuffle_indices]
shuffled_labels = labels[shuffle_indices]

imgs = shuffled_data[:N_LABLED+N_UNLABLED]
random_labels = shuffled_labels[:N_LABLED+N_UNLABLED]

#
# i=70
# k=10
#
# imgs[i] = np.ones((IMG_SIZE,IMG_SIZE))
# imgs[k] = np.ones((IMG_SIZE,IMG_SIZE))
#

# %% imports for signal enhancement

import sys
sys.path.append('/scratch/guysharon/Work/CryoEMSignalEnhancement')
from src.cryo_signal_enhance import *


# %%
starfile_fullpath = '/data/yoavharlap/saved_data/starfile.star'

writeable_dir = "/data/yoavharlap/saved_data"  # this is where data will be saved
CreateStarFileFromImages_WithoutCTFData(starfile_fullpath, imgs)

img_size = IMG_SIZE
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


warnings.filterwarnings(action="ignore", message="unclosed", category=ResourceWarning)
warnings.filterwarnings("ignore")  # I get an annoying socket warning

PrepareOpts(starfile_fullpath, writeable_dir, opts)
preprocessed_source = PreprocessData(starfile_fullpath, opts)
basis = FSPCA(preprocessed_source, opts)

# %% image lustering


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


for i in range(len(adj_matrix)):
    adj_matrix[i, i] = 0
plt.imshow(adj_matrix)
plt.title("correlations between first 50 members of the first class (diagonal removed)")
plt.show()
#
# i=70
# k=10
# print("The most collective  with i has a correlation:",max(adj_matrix[i][:]),"not",adj_matrix[i][k])
# fig, axarr = plt.subplots(1,2)
# str1 = "img: "+ str(i)
# axarr[0].set_title(str1)
# str2 = "img: "+ str(k)
# axarr[1].set_title(str2)
# axarr[0].imshow(imgs[i],cmap='gray')
# axarr[1].imshow(imgs[k],cmap='gray')
# title1 = "corr: "+ str(adj_matrix[i][k])
# fig.suptitle(title1, fontsize=16)
#
# c = np.argmax(adj_matrix[i][:], axis=0)
# fig, axarr = plt.subplots(1,2)
# str1 = "img: "+ str(i)
# axarr[0].set_title(str1)
# str2 = "img: "+ str(c)
# axarr[1].set_title(str2)
# axarr[0].imshow(imgs[i],cmap='gray')
# axarr[1].imshow(imgs[c],cmap='gray')
# title1 = "corr: "+ str(adj_matrix[i][c])
# fig.suptitle(title1, fontsize=16)
# plt.show()


from classification_functions import run_process,adj_matrix_analysis,visual_error_2,eror_sum1

n_labeled, n_unlabeled = N_LABLED, N_UNLABLED
# true_labels = make_threshold_0(random_labels)  # make 0 -1
true_labels = random_labels
labels, list_index_of_erors = run_process(n_labeled, n_unlabeled, true_labels, adj_matrix, preprocessed_source)
print("eror_sum1:",len(list_index_of_erors))
# # make_for_run(true_labels,n_labeled,n_unlabeled,adj_matrix)
# # visual_error(n_labeled, n_unlabeled, adj_matrix, labels,true_labels,imgs)
adj_matrix_analysis(adj_matrix, true_labels)
visual_error_2(n_labeled, n_unlabeled, adj_matrix, labels, true_labels, imgs)

