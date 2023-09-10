
from classification_functions import *

# from guy import *
import os

# prev_pwd = os.getcwd()
# os.chdir('/scratch/guysharon/Work/CryoEMSignalEnhancement')
# from src.cryo_signal_enhance import *


# work
import sys
sys.path.append('/scratch/guysharon/Work/CryoEMSignalEnhancement/src')

from cryo_signal_enhance import *

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import mrcfile
import matplotlib.pyplot as plt
import glob, os

images = []
outliers = []

good_images_filenames = []
outliers_filenames = []

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
        good_images_filenames.append(file)

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
        outliers_filenames.append(file)

import random

random.shuffle(images)
random.shuffle(outliers)

print(len(images))
print(len(outliers))

good_images = images


n_labeled, n_unlabeled = N_LABLED, N_UNLABLED
dates_path = "/home/yoavharlap/work/dates/dates.txt"

#p = 2 / 3
#random_labels = bernoulli.rvs(p, size=n_labeled + n_unlabeled)

j = 0


imgs = []
p = 2 / 3
random_labels = bernoulli.rvs(p, size=n_labeled + n_unlabeled)

k = 0
p = 0
for i in range(n_labeled + n_unlabeled):
    if (random_labels[i] == 0):
        
        random_labels[i] = -1
        imgs.append(outliers[k])
        k = k + 1
    else:
        imgs.append(good_images[p])
        p = p + 1

imgs = np.array(imgs)



#imgs[250] = np.ones((IMG_SIZE,IMG_SIZE))
#imgs[650] = np.ones((IMG_SIZE,IMG_SIZE))
#random_labels[250] = -1
#random_labels[650] = -1


# mrcs.set_data(imgs)
# mrcs.close()
# %% imports for signal enhancement

#
# import os
#
# prev_pwd = os.getcwd()
# os.chdir('/scratch/guysharon/Work/CryoEMSignalEnhancement')
# from src.cryo_signal_enhance import *

# os.chdir(prev_pwd)

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



# %% image clustering
opts["class_theta_step"] = 5
opts["num_classes"] = 3000
opts["num_classes"] = 1
opts["class_size"] = N_LABLED + N_UNLABLED
opts["skip_class_sort"] = False
opts['debug_verbose'] = False

opts["class_gpu"] = 0
classes, classes_ref, chosen_classes, original_chosen_classes, corrs = Classify_2D(basis, opts)
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
plt.title("correlations")
plt.show()

# %%

n_labeled, n_unlabeled = N_LABLED, N_UNLABLED
true_labels = make_threshold_0(random_labels)#make 0 -1
labels, list_index_of_erors = run_process(n_labeled, n_unlabeled, true_labels, adj_matrix, preprocessed_source)
# make_for_run(true_labels,n_labeled,n_unlabeled,adj_matrix)
#visual_error(n_labeled, n_unlabeled, adj_matrix, labels,true_labels,imgs)
adj_matrix_analysis(adj_matrix, true_labels)
visual_error_2(n_labeled, n_unlabeled, adj_matrix, labels,true_labels,imgs)
   
