import numpy as np
import matplotlib.pyplot as plt
from k_nn_functions import *

n_labeled, n_unlabeled = 1200, 300
step_size = 100  # Adjust this step size as needed

# Paths to numpy cryo arrays
outliers_file_path = "/data/yoavharlap/10028_classification/outliers_images.npy"
particles_file_path = "/data/yoavharlap/10028_classification/particles_images.npy"

outliers_data = np.load(outliers_file_path)
particles_data = np.load(particles_file_path)
data = np.concatenate((outliers_data, particles_data), axis=0)
labels = np.concatenate((np.ones(len(outliers_data)), np.zeros(len(particles_data))))

# Create an index array to shuffle data and labels in the same way
shuffle_indices = np.arange(len(data))

# Shuffle data and labels using the shuffled indices
np.random.shuffle(shuffle_indices)
imgs = data[shuffle_indices]
true_labels = labels[shuffle_indices]

imgs = imgs[:n_labeled+n_unlabeled]
true_labels = true_labels[:n_labeled+n_unlabeled]
total_samples = len(true_labels)

filtered_images = np.empty_like(imgs)
sigma = 0
for i in range(len(imgs)):
    filtered_image = ndimage.gaussian_filter(imgs[i], sigma=sigma)
    filtered_images[i] = filtered_image

print("filtered done")
imgs = filtered_images
# Calculate the adjacency matrix using all data
adj_matrix = calc_similarity_matrix(imgs)

# Initialize arrays to store the sum of errors for each iteration
sum_errors_nn_list = []
sum_errors_nn2_list = []
sum_errors_nn_10_list = []  # Initialize for 10-NN


# Initialize an array to keep track of which samples are labeled
is_labeled = np.zeros(total_samples, dtype=bool)
steps_num = 10
num_labeled_array = np.linspace(10, n_labeled, steps_num).astype(int)

for num_labeled in num_labeled_array:
    total_samples_2 = num_labeled + n_unlabeled
    # Determine which samples to label in this iteration
    num_new_labeled = num_labeled - np.sum(is_labeled[:num_labeled])
    labeled_indices = np.where(~is_labeled)[0][:num_new_labeled]

    # Update the is_labeled array to mark the newly labeled samples
    is_labeled[labeled_indices] = True

    labels = np.zeros(total_samples)
    labels[is_labeled] = true_labels[is_labeled]

    #nearest_neighbor_labels = nearest_neighbor_from_labeled(labels, adj_matrix, num_labeled, n_unlabeled)
    #errors_indexes_nn = np.where(np.not_equal(nearest_neighbor_labels[num_labeled:], true_labels[num_labeled:]))

    #labels2 = np.zeros(total_samples)
    #labels2[is_labeled] = true_labels[is_labeled]
    
    nearest_neighbor_labels = nearest_neighbor(labels, adj_matrix, num_labeled, n_unlabeled)
    
    #nearest_neighbor_labels2 = np.concatenate((np.zeros(num_labeled),nearest_neighbor_labels2))
    
    errors_indexes_nn2 = np.where(np.not_equal(nearest_neighbor_labels, true_labels[n_labeled:]))

    # Calculate the sum of errors for both approaches
    #sum_errors_nn = len(errors_indexes_nn[0])
    sum_errors_nn2 = len(errors_indexes_nn2[0])

    # Append the sum of errors to the respective lists
    #sum_errors_nn_list.append(sum_errors_nn)
    sum_errors_nn2_list.append(sum_errors_nn2)
    # Calculate nearest neighbor labels using 10-NN approach
    nearest_neighbor_labels_10 = nearest_neighbor_10(labels, adj_matrix, num_labeled,n_labeled, n_unlabeled)

    # Concatenate zeros to match sizes
    #nearest_neighbor_labels_10 = np.concatenate((np.zeros(num_labeled), nearest_neighbor_labels_10))
    
    
    errors_indexes_nn_10 = np.where(np.not_equal(nearest_neighbor_labels_10, true_labels[n_labeled:]))
    
    
    # Calculate the sum of errors for 10-NN approach
    sum_errors_nn_10 = len(errors_indexes_nn_10[0])
    
    # Append the sum of errors to the list for 10-NN
    sum_errors_nn_10_list.append(sum_errors_nn_10)

    
   
    
    
# Plot the sum of errors for each approach
plt.plot(num_labeled_array, sum_errors_nn2_list, label="Nearest Neighbor")
plt.plot(num_labeled_array, sum_errors_nn_10_list, label="10-NN")
plt.xlabel("Number of Labeled Samples")
plt.ylabel("Sum of Errors")
plt.legend()
plt.show()

labels = np.concatenate((true_labels[:n_labeled],nearest_neighbor_labels))
visual_error_2(n_labeled, n_unlabeled, adj_matrix, labels, true_labels, imgs)
