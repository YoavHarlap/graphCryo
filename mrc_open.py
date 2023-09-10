
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

for i in range(50):
    plt.imshow(outliers[i],cmap='gray')
    plt.show()
    
    