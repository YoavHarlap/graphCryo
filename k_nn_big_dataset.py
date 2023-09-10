from k_nn_functions import *

n_labeled, n_unlabeled = 1200, 300
#
# good_imgs_path = "/data/yoavharlap/eman_particles/good"
# outliers_imgs_path = "/data/yoavharlap/eman_particles/outliers"
# good_imgs = load_mrcs_from_path_to_arr(good_imgs_path)
# outliers_imgs = load_mrcs_from_path_to_arr(outliers_imgs_path)
#
# random.shuffle(good_imgs)
# random.shuffle(outliers_imgs)
#
# print(len(good_imgs))
# print(len(outliers_imgs))
# p = 2 / 3
# random_labels = bernoulli.rvs(p, size=n_labeled + n_unlabeled)
# # random_labels = np.array([0] * len(outliers_imgs) + [1] * len(good_imgs#))
# # random.shuffle(random_labels)
# imgs, true_labels = make_imgs_arr_from_labels(random_labels, good_imgs, outliers_imgs)


# Paths to your numpy files
outliers_file_path = "/data/yoavharlap/10028_classification/outliers_images.npy"
particles_file_path = "/data/yoavharlap/10028_classification/particles_images.npy"

outliers_data = np.load(outliers_file_path)
particles_data = np.load(particles_file_path)
data = np.concatenate((outliers_data, particles_data), axis=0)
labels = np.concatenate((np.ones(len(outliers_data)), np.zeros(len(particles_data))))
train_ratio = 0.8
total_samples = len(labels)
train_samples = int(train_ratio * total_samples)

# Create an index array to shuffle data and labels in the same way
shuffle_indices = np.arange(len(data))
np.random.shuffle(shuffle_indices)

# Shuffle data and labels using the shuffled indices
imgs = data[shuffle_indices]
imgs = imgs[:n_labeled+n_unlabeled]
true_labels = labels[shuffle_indices]
true_labels = true_labels[:n_labeled+n_unlabeled]


imgs_copy = imgs.copy()
from scipy import ndimage

gaussian_numbers = [0,10,15,20,25,30]
for gaussian_num in gaussian_numbers:

    if (gaussian_num != 0):
        for i, img in enumerate(imgs_copy):
            imgs[i] = ndimage.gaussian_filter(img, gaussian_num)
    else:
        imgs = imgs_copy.copy()
    # Guy's correlations
    adj_matrix = calc_similarity_matrix(imgs)
    # print("done")

    labels = np.zeros(n_labeled + n_unlabeled)
    labels[:n_labeled] = true_labels[:n_labeled]

    steps = np.linspace(15, n_labeled, 10)
    labels_save = np.array(labels)
    adj_matrix_save = np.array(adj_matrix)
    n_labeled_save = n_labeled
    nn_errors = []
    nn_2_errors = []
    na_errors = []
    no_errors = []
    knn_errors = []
    na_errors_2 = []
    for n_labeled in steps:
        n_labeled = int(n_labeled)
        labels = labels_save[n_labeled_save - n_labeled: n_unlabeled + n_labeled_save]
        adj_matrix = adj_matrix_save[n_labeled_save - n_labeled: n_unlabeled + n_labeled_save,
                     n_labeled_save - n_labeled: n_unlabeled + n_labeled_save]

        if (is_square(adj_matrix) == False):
            print("truuuuuuuuuuuueeeeeeeeeeeeeeeee")

        nearest_neighbor_labels = nearest_neighbor_from_labeled(labels, adj_matrix, n_labeled, n_unlabeled)
        errors_indexes_nn = np.where(np.not_equal(nearest_neighbor_labels, true_labels[
                                                                           n_labeled_save - n_labeled: n_unlabeled + n_labeled_save]))
        sum_of_errors_nn = np.size(errors_indexes_nn, 1)
        nn_errors.append(sum_of_errors_nn)
        # print("nn: errors_indexes_nearest_neighbor:",errors_indexes_nn)
        # print("nn: errors_sum_nearest_neighbor:",sum_of_errors_nn)

        k = 10

        knn_labels, new_algo_labels = get_labels_median_k_nn(labels, k, adj_matrix, n_labeled, n_unlabeled)
        errors_indexes_knn = np.where(
            np.not_equal(knn_labels, true_labels[n_labeled_save - n_labeled: n_unlabeled + n_labeled_save]))
        sum_of_errors_knn = np.size(errors_indexes_knn, 1)
        knn_errors.append(sum_of_errors_knn)
        # print("knn: errors_indexes_nearest_neighbor:",errors_indexes_knn)
        # print("knn: errors_sum_nearest_neighbor:",sum_of_errors_knn)

        errors_indexes_na = np.where(
            np.not_equal(new_algo_labels, true_labels[n_labeled_save - n_labeled: n_unlabeled + n_labeled_save]))
        sum_of_errors_na = np.size(errors_indexes_na, 1)
        na_errors.append(sum_of_errors_na)
        # print("na: errors_indexes_nearest_neighbor:",errors_indexes_na)
        # print("na: errors_sum_nearest_neighbor:",sum_of_errors_na)

        n_outliers = 4
        n_out_labels = get_labels_n_outliers_from_k_nn(labels, n_outliers, k, adj_matrix, n_labeled, n_unlabeled)
        errors_indexes_no = np.where(
            np.not_equal(n_out_labels, true_labels[n_labeled_save - n_labeled: n_unlabeled + n_labeled_save]))
        sum_of_errors_no = np.size(errors_indexes_no, 1)
        no_errors.append(sum_of_errors_no)
        # print("no: errors_indexes_nearest_neighbor:",errors_indexes_no)
        # print("no: errors_sum_nearest_neighbor:",sum_of_errors_no)

        # print("na: errors_indexes_nearest_neighbor:",errors_indexes_na)
        # print("na: errors_sum_nearest_neighbor:",sum_of_errors_na)

        # new_algo_labels_2 = get_labels_corr_threshold(labels,k,adj_matrix, n_labeled, n_unlabeled)
        # errors_indexes_na_2 = np.where(np.not_equal(new_algo_labels_2, true_labels[n_labeled_save -n_labeled: n_unlabeled+n_labeled_save]))
        # sum_of_errors_na_2 = np.size(errors_indexes_na_2,1)
        # na_errors_2.append(sum_of_errors_na_2)
        # print("na_2: errors_indexes_nearest_neighbor_2:",errors_indexes_na_2)
        # print("na_2: errors_sum_nearest_neighbor_2:",sum_of_errors_na_2)

        min_errors = [sum_of_errors_nn, sum_of_errors_knn, sum_of_errors_na, sum_of_errors_no]

        print("gaussian_num:", gaussian_num)
        print("n_labeled:", n_labeled)
        print(np.min(min_errors))
        print(np.argmin(min_errors))

    # labels1, list_index_of_erors1 = run_process(n_labeled, n_unlabeled, true_labels, adj_matrix, 0)

    n_labeled = n_labeled_save
    labels = labels_save
    # visual_error_2(n_labeled, n_unlabeled, adj_matrix, new_algo_labels,true_labels,imgs)

    plt.plot(steps, nn_errors, '-o', label="Nearest Neighbor")
    plt.plot(steps, knn_errors, '-o', label="K Nearest Neighbor")
    plt.plot(steps, na_errors, '->', label="new_algo")
    plt.plot(steps, no_errors, '-^', label="4 outliers in NN as outlier")
    # plt.plot(steps,na_errors_2,'-o', label="new_algo_2")

    plt.legend()

    title = "errors: " + str(np.min(min_errors))
    plt.xlabel("num of labeled data")
    plt.ylabel("num of errors")
    plt.title(title)
    plt.show()

    print("lololol")
    #calc_sum_corr(adj_matrix, true_labels)

# im = outliers_imgs[i]
# fig, ax = plt.subplots(1,2,figsize=(15,15))
# ax[0].imshow(im, cmap='gray')
# ax[0].set_title('original: outlier', fontsize = f_size)
# ax[1].imshow(ndimage.gaussian_filter(im, 20), cmap = 'gray')
# ax[1].set_title('gaussian filter 20', fontsize = f_size);
# plt.show()
# i = i+1
