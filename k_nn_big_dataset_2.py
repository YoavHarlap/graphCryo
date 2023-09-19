import numpy as np
import matplotlib.pyplot as plt
from k_nn_functions import *  # Import necessary functions
from scipy import ndimage

# Set the number of labeled and unlabeled samples
n_labeled, n_unlabeled = 1200, 300

# Step size for the number of labeled samples
step_size = 100  # Adjust this step size as needed

# Paths to numpy cryo arrays
outliers_file_path = "/data/yoavharlap/10028_classification/outliers_images.npy"
particles_file_path = "/data/yoavharlap/10028_classification/particles_images.npy"

outliers_data = np.load(outliers_file_path)
particles_data = np.load(particles_file_path)

# Concatenate the data and labels
data = np.concatenate((outliers_data, particles_data), axis=0)
labels = np.concatenate((np.ones(len(outliers_data)), np.zeros(len(particles_data))))

# Shuffle data and labels
shuffle_indices = np.arange(len(data))
np.random.shuffle(shuffle_indices)
imgs = data[shuffle_indices]
true_labels = labels[shuffle_indices]

# Select a subset of data and labels
imgs = imgs[:n_labeled + n_unlabeled]
true_labels = true_labels[:n_labeled + n_unlabeled]
total_samples = len(true_labels)

# Apply Gaussian filtering to the images
filtered_images = np.empty_like(imgs)
sigma = 20
for i in range(len(imgs)):
    filtered_image = ndimage.gaussian_filter(imgs[i], sigma=sigma)
    filtered_images[i] = filtered_image

print("Gaussian filtering done")
imgs = filtered_images

# Calculate the adjacency matrix using all data
adj_matrix = calc_similarity_matrix(imgs)

# Initialize arrays to store the sum of errors for each iteration
sum_errors_nn2_list = []
sum_errors_nn_10_lists = [[] for _ in range(11)]  # Initialize for 10-NN for each x

# Initialize an array to keep track of which samples are labeled
is_labeled = np.zeros(total_samples, dtype=bool)
steps_num = 10
num_labeled_array = np.linspace(10, n_labeled, steps_num).astype(int)

for num_labeled in num_labeled_array[:]:
    total_samples_2 = num_labeled + n_unlabeled
    # Determine which samples to label in this iteration
    num_new_labeled = num_labeled - np.sum(is_labeled[:num_labeled])
    labeled_indices = np.where(~is_labeled)[0][:num_new_labeled]

    # Update the is_labeled array to mark the newly labeled samples
    is_labeled[labeled_indices] = True

    labels = np.empty(total_samples)
    labels[is_labeled] = true_labels[is_labeled]

    for x in range(11):
        # Calculate nearest neighbor labels using 10-NN approach for the current x
        nearest_neighbor_labels_10 = nearest_neighbor_10(labels, adj_matrix, num_labeled, n_labeled, n_unlabeled, true_labels, x)
        errors_indexes_nn_10 = np.where(np.not_equal(nearest_neighbor_labels_10, true_labels[n_labeled:]))

        # Calculate the sum of errors for 10-NN approach for the current x
        sum_errors_nn_10 = len(errors_indexes_nn_10[0])

        # Append the sum of errors to the list for the current x
        sum_errors_nn_10_lists[x].append(sum_errors_nn_10)

# Plot the sum of errors for each approach for different x values
plt.figure(figsize=(10, 6))

# Plot sum of errors for Nearest Neighbor approach
#plt.plot(num_labeled_array, sum_errors_nn_2_lists, label="Nearest Neighbor")

# Plot sum of errors for 10-NN approach for each x
for x in range(11):
    plt.plot(num_labeled_array, sum_errors_nn_10_lists[x], label=f"10-NN (x={x})")

plt.xlabel("Number of Labeled Samples")
plt.ylabel("Sum of Errors")
plt.title(f"Sum of Errors for Nearest Neighbor and 10-NN for Different x\nSigma: {sigma}")
plt.legend()
plt.grid(True)
plt.show()
