from classification_functions import *



import time
CURRENT_TIME = time.strftime("%Y-%m-%d_%H:%M:%S")

# Print the current date and time
print("The current date and time is:", CURRENT_TIME)

logger = logging.getLogger(__name__)

# %%
writeable_dir = WRITEABLE_DIR
os.chdir(writeable_dir)
og_v = Volume.load("/scratch/guysharon/Work/datafiles/volrefs/emd_2660.map", dtype=np.float64)
logger.info("Original volume map data" f" shape: {og_v.shape} dtype:{og_v.dtype}")

img_size = IMG_SIZE
#img_size = 28


v = og_v.downsample(img_size)
L = v.resolution
n_labeled, n_unlabeled = N_LABLED, N_UNLABLED
num_imgs = n_labeled+n_unlabeled  # How many images in our source.
#noise_variance = 5e-4  # Set a target noise variance
noise_variance = NOISE_VAR_CRYO  # Set a target noise variance

print(num_imgs)

# %%


# %%

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

save_path = "/home/yoavharlap/work/" + CURRENT_TIME
os.mkdir(save_path)
save_path = save_path + "/starfile.star"
tmp = src.save(save_path, overwrite=True, batch_size=150000)
#tmp = src.save("/home/yoavharlap/work/starfile1.star", overwrite=True, batch_size=150000)


dates_path = "/home/yoavharlap/work/dates/dates.txt" 



file = open(dates_path, "a")
string = '\n' + CURRENT_TIME+'#'+str(tmp)
a = file.write(string)
file.close()
print(a)

