
from k_nn_functions import *

n_labeled , n_unlabeled = 300,600

good_imgs_path  = "/data/yoavharlap/eman_particles/good"
outliers_imgs_path = "/data/yoavharlap/eman_particles/outliers"
good_imgs = load_mrcs_from_path_to_arr(good_imgs_path)
outliers_imgs = load_mrcs_from_path_to_arr(outliers_imgs_path)

random.shuffle(good_imgs)
random.shuffle(outliers_imgs)

print(len(good_imgs))
print(len(outliers_imgs))

#for i in range(10):
#    dark_image_grey = outliers_imgs[i]
#    dark_image_grey_fourier = np.fft.fftshift(np.fft.fft2(dark_image_grey))
 #   plt.imshow(np.log(abs(dark_image_grey_fourier)), cmap='gray')
 #   plt.show()
    
    
    
    
    
def fourier_masker_ver(image, i):
    f_size = 15
    dark_image_grey_fourier = np.fft.fftshift(np.fft.fft2(image))
    dark_image_grey_fourier[:225, 235:240] = i
    dark_image_grey_fourier[-225:,235:240] = i
    fig, ax = plt.subplots(1,3,figsize=(15,15))
    ax[0].imshow(np.log(abs(dark_image_grey_fourier)), cmap='gray')
    ax[0].set_title('Masked Fourier', fontsize = f_size)
    ax[1].imshow(image, cmap = 'gray')
    ax[1].set_title('Greyscale Image', fontsize = f_size);
    ax[2].imshow(abs(np.fft.ifft2(dark_image_grey_fourier)), 
                     cmap='gray')
    ax[2].set_title('Transformed Greyscale Image', 
                     fontsize = f_size);

dark_image = good_imgs[0]   
index = 10 
fourier_masker_ver(dark_image, index)


def fourier_masker_hor(image, i):
    f_size = 15
    dark_image_grey_fourier = np.fft.fftshift(np.fft.fft2(image))
    dark_image_grey_fourier[235:240, :230] = i
    dark_image_grey_fourier[235:240,-230:] = i
    fig, ax = plt.subplots(1,3,figsize=(15,15))
    ax[0].imshow(np.log(abs(dark_image_grey_fourier)), cmap='gray')
    ax[0].set_title('Masked Fourier', fontsize = f_size)
    ax[1].imshow(image, cmap = 'gray')
    ax[1].set_title('Greyscale Image', fontsize = f_size);
    ax[2].imshow(abs(np.fft.ifft2(dark_image_grey_fourier)), 
                     cmap='gray')
    ax[2].set_title('Transformed Greyscale Image', 
                     fontsize = f_size);

fourier_masker_hor(dark_image, index)

def fourier_iterator(image, value_list):
    for i in value_list:
        fourier_masker_ver(image, i)
 
fourier_iterator(dark_image, [0.001, 1, 100])


