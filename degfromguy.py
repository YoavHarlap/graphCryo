import logging
print("hii")
import pyfftw
import matplotlib.pyplot as plt
import numpy as np
import cupy as cp

NOISE_VAR_CRYO = 0
NOISE_VAR_OUTLIER = 0
CRYONUMIMAGES = 1300
N_LABLED = 400
N_UNLABLED = 1001
SHOW = True
from aspire.abinitio import CLSyncVoting
from aspire.basis import FFBBasis2D, FFBBasis3D
from aspire.classification import BFSReddyChatterjiAverager2D, RIRClass2D
from aspire.denoising import DenoiserCov2D
from aspire.noise import AnisotropicNoiseEstimator, CustomNoiseAdder
from aspire.operators import FunctionFilter, RadialCTFFilter
from aspire.reconstruction import MeanEstimator
from aspire.source import Simulation, RelionSource, ArrayImageSource
from aspire.source.simulation import randn
from aspire.basis.fspca import FSPCABasis
from aspire.storage import StarFile
from aspire.utils.coor_trans import (
    get_aligned_rotations,
    get_rots_mse,
    register_rotations,
)
from aspire.volume import Volume
import os
from aspire.source import RelionSource, Simulation

logger = logging.getLogger(__name__)

# %%
num_imgs = N_LABLED+N_UNLABLED  # How many images in our source.
noise_variance = 5e-4  # Set a target noise variance
noise_variance = NOISE_VAR_CRYO  # Set a target noise variance
img_size = 189
img_size = 28

writeable_dir = "/data/yoavharlap/saved_data"  # this is where data will be saved
print(num_imgs)


# %%
os.chdir(writeable_dir)
og_v = Volume.load("/scratch/guysharon/Work/datafiles/volrefs/emd_2660.map", dtype=np.float64)
logger.info("Original volume map data" f" shape: {og_v.shape} dtype:{og_v.dtype}")
v = og_v.downsample(img_size)
L = v.resolution


# Then create a filter based on that variance
# This is an example of a custom noise profile
def noise_function(x, y):
    alpha = 1
    beta = 1
    # White
    f1 = noise_variance
    # Violet-ish
    f2 = noise_variance * (x * x + y * y) / L * L
    return (alpha * f1 + beta * f2) / 2.0

# %%
#functions
def make_threshold_0(list):
    th_list = [1 if list[v] > 0 else -1 for v in range(len(list))]
    return th_list

def eror_sum(list, true_labels):
    th_list = make_threshold_0(list)
    erors = [0 if th_list[v] == true_labels[v] else 1 for v in range(len(th_list))]
    sums_erors = np.array(erors).sum(0)
    list_index_of_erors = []
    erors = [list_index_of_erors.append(v) if erors[v] == 1 else 1 for v in range(len(th_list))]
    return sums_erors,list_index_of_erors

def eror_sum1(list, true_labels):
    th_list = make_threshold_0(list)
    erors = [0 if th_list[v] == true_labels[v] else 1 for v in range(len(th_list))]
    sums_erors = np.array(erors).sum(0)
    list_index_of_erors = []
    return sums_erors

def run_process(n_labeled, n_unlabeled,true_labels, adj_matrix,preprocessed_source):
    make_for_run(true_labels,n_labeled,n_unlabeled,adj_matrix)


    labels = np.zeros(n_unlabeled)
    labels = np.concatenate((true_labels[0:n_labeled], labels))
    start = init_nearest_labels(n_labeled, n_unlabeled, adj_matrix, labels)
    #print("error sum: ", eror_sum(start, true_labels))
    #print("start: \n", start)
    #print("true labels: \n", true_labels)
    #print()
   # fig = plt.figure(figsize=(28, 28))
    j = 0
    arr = true_labels
    arr = start
    for i in range(20):
        if arr[i] == 1:
            j = j + 1
            print("hi0")
            #im = data[i]
            #fig.add_subplot(28, 28, j)
            if SHOW:
                preprocessed_source.images(i,1).show()  
      #plt.show()
    j = 0
   # fig = plt.figure(figsize=(28, 28))
    for i in range(20):
        if arr[i] == -1:
            j = j + 1
            print("hi1")
            plt.show()
            #im = data[i]
           # fig.add_subplot(28, 28, j)
            if SHOW:
                preprocessed_source.images(i,1).show()  
    #plt.show()
    eror_sum1,list_index_of_erors = eror_sum(start, true_labels)
    print("error sum: ",  eror_sum1)
    #print("list index errors",list_index_of_erors)





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
    return labels1



# %%
custom_noise = CustomNoiseAdder(noise_filter=FunctionFilter(noise_function))

logger.info("Initialize CTF filters.")
# Create some CTF effects
pixel_size = 5 * 65 / img_size  # Pixel size of the images (in angstroms)
voltage = 200  # Voltage (in KV)
defocus_min = 1.5e4  # Minimum defocus value (in angstroms)
defocus_max = 2.5e4  # Maximum defocus value (in angstroms)
defocus_ct = 7  # Number of defocus groups.
Cs = 2.0  # Spherical aberration
alpha = 0.1  # Amplitude contrast

# Create filters
ctf_filters = [
    RadialCTFFilter(pixel_size, voltage, Cs=2.0, alpha=0.1)
                 ]                                                                              
# Finally create the Simulation

src = Simulation(
L=v.resolution,
    n=num_imgs,
    vols=v,
    noise_adder=custom_noise,
    unique_filters=ctf_filters,
    dtype=v.dtype,
    offsets = img_size / 16 * randn(2, num_imgs).astype(float).T / 50
)

#%% add pixel size to metadata
src._metadata.insert(0,'_rlnPixelSize', pixel_size)

#%% create starfile and mrcs

tmp = src.save("/home/yoavharlap/work/starfile.star", overwrite=True, batch_size=150000)
starfile_fullpath = tmp['starfile']
mrcs_fullpath = os.path.join(os.path.dirname(tmp['starfile']),tmp['mrcs'][0])
import mrcfile
mrcs = mrcfile.open(mrcs_fullpath,'r+')
imgs = np.array(mrcs.data)

#
# manipulate images, for example 'imgs[0] = np.zeros((189,189))'
#



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

#%% add fake images

n_labeled, n_unlabeled = N_LABLED, N_UNLABLED
from matplotlib.image import imread
from scipy.stats import bernoulli
p = 2/3
random_labels = bernoulli.rvs(p, size=n_labeled+ n_unlabeled)
a = np.zeros(n_labeled+ n_unlabeled-51)
b = np.ones(51)
#random_labels = np.concatenate((b, a), axis=0)

dataset_name = 'zero_mnist'
dataset_path = "/data/yoavharlap/" + dataset_name
filenames = os.listdir(dataset_path)
#random.shuffle(filenames)
print('dataset path:', dataset_path)
j = 0
for i in range(n_labeled+ n_unlabeled):
       if (random_labels[i] == 0):
           random_labels[i] = -1 #for classification
           #filename = filenames[j]
           #print(j, ":", filename)
           #category = filename.split('.')[0]
           #file_path = dataset_path + '/{}'.format(filename)
           #img = imread(file_path)
           #img = imread("/home/yoavharlap/work/outlire3.png")
           #img = img[:, :, 0]
           #plt.imshow(img)
           #plt.show()
            # img0 = np.expand_dims(img, axis=0) # make(28,28) to (1,28,28) like cryo
          # img1 = img0.astype('float64')
          # print("im1",img1.shape)
           #print("src",src.images[i]._data.shape)
           #imgs[i] = noisy(img)
           #imgs[i] = img
           imgs[i] = np.ones((img_size,img_size))

          # print(img1) 
           j = j + 1
                      



mrcs.set_data(imgs)
mrcs.close()
#%% imports for signal enhancement
prev_pwd = os.getcwd()
os.chdir('/scratch/guysharon/Work/CryoEMSignalEnhancement')
from src.cryo_signal_enhance import *
os.chdir(prev_pwd)
import matplotlib.pyplot as plt
import pickle

#%% options for signal enhancement

opts = {
    #   GENERAL
        "N":                    2147483647,
        'verbose':              True,


    #   PREPROCESSING
       # "downsample":           89,
        "downsample":           28,
        "batch_size":           2**15,
        "num_coeffs":           1500,

    #   EM
        "ctf":                  True,
        "iter":                 7,
        "norm":                 True,
        "em_num_inputs":        150,
        "take_last_iter":       True,
        "em_par":               True,
        "em_gpu":               False,
        "num_class_avg":        1500,
        "num_class_avg":        1,

    #   DEBUG
        'debug_verbose':        True,
        'random_seed':          -1,
        'sample':               False,
    }

opts['preprocess_flags'] = [1,1,0,0,0]


warnings.filterwarnings(action="ignore", message="unclosed", category=ResourceWarning)
warnings.filterwarnings("ignore") # I get an annoying socket warning
PrepareOpts(starfile_fullpath, writeable_dir, opts)
preprocessed_source = PreprocessData(starfile_fullpath,opts)




basis = FSPCA(preprocessed_source,opts)


#%% image clustering
opts["class_theta_step"]    = 5
opts["num_classes"]         = 3000
opts["num_classes"]         = 1
opts["class_size"]          = N_LABLED+N_UNLABLED
opts["class_gpu"]           = 0
opts["skip_class_sort"]     = False
opts['debug_verbose']       = False

classes, classes_ref, chosen_classes, original_chosen_classes, corrs = Classify_2D(basis, opts)

#%%
#print(f"the correlation between the image {classes[0,4]} and {classes[1,4]} is {corrs[4,0,1]}")
#print(f"the correlation between the image {classes[3,15]} and {classes[4,15]} is {corrs[15,3,4]}")
#print("the correlation between the image 'classes[i,k]' and 'classes[j,k] is 'corrs[k,i,j]'")

from matplotlib import pyplot as plt
data = corrs[0][0:n_labeled+n_unlabeled,0:n_labeled+n_unlabeled]
for i in range(len(data)):
    data[i,i] = 0
#plt.imshow(data)
#plt.title("correlations between first 50 members of the first class (diagonal removed)")
#plt.show()


#%%
adj_matrix = np.zeros((N_LABLED+N_UNLABLED,N_LABLED+N_UNLABLED))
for i in range(len(classes[:, 0])):
    print("hi ",i)
    for j in range(i+1,len(classes[:, 0])):
        imgi = classes[i,0]
        imgj = classes[j,0]
        corr = 100*corrs[0,i,j]
        adj_matrix[imgi][imgj] = corr
        adj_matrix[imgj][imgi] = corr
        
        #print("the correlation between the image 'classes[i,0]' and 'classes[j,0] is 'corrs[0,i,j]'")
        #print(imgi,imgj,corr)
        
        

for i in range(len(adj_matrix)):
    adj_matrix[i,i] = 0    
plt.imshow(adj_matrix)
plt.title("correlations between first 50 members of the first class (diagonal removed)")
plt.show()
    
    
    
#%%
n_labeled, n_unlabeled = N_LABLED, N_UNLABLED
true_labels = random_labels
#adj_matrix = data
run_process(n_labeled, n_unlabeled,true_labels, adj_matrix,preprocessed_source)

#make_for_run(true_labels,n_labeled,n_unlabeled,adj_matrix)
