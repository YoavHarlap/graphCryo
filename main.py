import logging
print("hello")
import matplotlib.pyplot as plt
import numpy as np

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

logger = logging.getLogger(__name__)

#%%
num_imgs = 50000  # How many images in our source.
noise_variance = 5e-4  # Set a target noise variance
img_size = 189

#%%
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
    RadialCTFFilter(pixel_size, voltage, defocus=d, Cs=2.0, alpha=0.1)
    for d in np.linspace(defocus_min, defocus_max, defocus_ct)
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
tmp = src.save("/scratch/guysharon/Work/Python/saved_test_data/starfile.star", overwrite=True, batch_size = 150000)
starfile_fullpath = tmp['starfile']

#%%
prev_pwd = os.getcwd()
os.chdir('/scratch/guysharon/Work/CryoEMSignalEnhancement')
from src.cryo_signal_enhance import *
os.chdir(prev_pwd)
import matplotlib.pyplot as plt
import pickle

writeable_dir = "/scratch/guysharon/Work/Python/saved_test_data"

opts = {
    #   GENERAL
        "N":                    2147483647,
        'verbose':              True,

    #   PREPROCESSING
        "downsample":           89,
        "batch_size":           2**15,
        "num_coeffs":           500,

    #   2D CLASS
        "class_theta_step":     5,
        "num_classes":          1500,
        "class_size":           300,
        "class_gpu":            0,

    #   EM
        "ctf":                  True,
        "iter":                 7,
        "norm":                 True,
        "em_num_inputs":        150,
        "take_last_iter":       True,
        "em_par":               True,
        "em_gpu":               False,
        "num_class_avg":        1500,

    #   DEBUG
        'debug_verbose':        True,
        'random_seed':          -1,
        'sample':               False
    }


warnings.filterwarnings(action="ignore", message="unclosed", category=ResourceWarning)
if ('timings' not in globals()): timings = {}
warnings.filterwarnings("ignore") # I get an annoying socket warning
PrepareOpts(starfile_fullpath, writeable_dir, opts)
preprocessed_source = PreprocessData(starfile_fullpath,opts)
basis = FSPCA(preprocessed_source,opts)
classes, classes_ref, chosen_classes, original_chosen_classes = Classify_2D(basis, opts)
rots, _ = calcRotsFromStarfile(basis.starfile)
degs = []
for i in range(100):
    coses = cosBetweenViewDirs(rots[:,:,classes[0,i]],rots[0:300,:,classes[:,i]])
    coses[abs(coses) > 1] = 1
    for j in range(len(coses)):
        degs.append(180 / np.pi * np.arccos(coses[j]))
n, bins, patches = plt.hist(x=degs, bins='auto', color='#0504aa', rwidth=1)
plt.xlim(xmax = 25, xmin = 0)
plt.xlabel('angular distance [deg]')
plt.ylabel('Frequency')
plt.title('angular distance distribution')

preprocessed_source.images(1,10).show()
                                                                                                                                                                                                               145,1         Bot
