from k_nn_functions import *

n_labeled , n_unlabeled = 1200,300
#pp = 200 20 20
good_imgs_path  = "/data/yoavharlap/eman_particles/good"
outliers_imgs_path = "/data/yoavharlap/eman_particles/outliers"
good_imgs = load_mrcs_from_path_to_arr(good_imgs_path)
outliers_imgs = load_mrcs_from_path_to_arr(outliers_imgs_path)

#random.shuffle(good_imgs)
#random.shuffle(outliers_imgs)

print(len(good_imgs))
print(len(outliers_imgs))
p = 2/3
#random_labels = bernoulli.rvs(p, size = n_labeled + n_unlabeled)
random_labels = np.array([-1, -1,  1,  1,  1, -1,  1,  1, -1,  1,  1,  1,  1,  1, -1, -1, -1,
        1, -1,  1, -1, -1,  1,  1,  1, -1,  1,  1, -1, -1,  1,  1,  1,  1,
        1, -1,  1,  1, -1,  1])
random_labels = bernoulli.rvs(p, size = n_labeled + n_unlabeled)

#random_labels = np.array([0] * len(outliers_imgs) + [1] * len(good_imgs#))
#random.shuffle(random_labels)     
#imgs,true_labels = make_imgs_arr_from_labels(random_labels, good_imgs[pp-n_labeled-n_labeled:pp], outliers_imgs)
imgs,true_labels = make_imgs_arr_from_labels(random_labels, good_imgs, outliers_imgs)

imgs_copy = imgs.copy()
from scipy import ndimage
true_labels_copy = true_labels.copy()
new_true_labels = true_labels.copy()
n_unlabeled_copy1 = n_unlabeled
gaussian_numbers = [0,20]
list_of_removed = []
for j,gaussian_num in enumerate(gaussian_numbers):
    
    if(j==1 or j==3):
        if(are_outliers (true_labels_copy,list_of_removed)):
            wrong_removed = False
        else:
            wrong_removed = True

        imgs = new_imgs.copy()
        true_labels = new_true_labels
        n_unlabeled = new_n_unlabeled 
        
        
    else:
        wrong_removed = False
        imgs = imgs_copy.copy()
        n_unlabeled = n_unlabeled_copy1
        true_labels = true_labels_copy
    if (gaussian_num!=0) :
        for i,img in enumerate(imgs):
            
            if(j!=10):
                
                   #img = normalize_img(img)
                print(i,"/" ,len(imgs_copy)-1)
                imgs[i] = ndimage.gaussian_filter(img, gaussian_num)
           
            if(j==90):
                #imgs[i] = img_rotate_avr(imgs[i])
                imgs[i] = normalize_img(imgs[i])
                imgs[i] = img_rotate_avr(imgs[i])
                imgs[i] = ndimage.median_filter(imgs[i],100)
    
    
        
        
    #Guy's correlations
    adj_matrix = calc_similarity_matrix(imgs)
    #print("done")
    if j==0:
        org_adj_matrix = adj_matrix.copy()
        new_imgs,list_of_removed = remove_corr_data(adj_matrix,n_labeled,n_unlabeled,imgs)
        new_n_unlabeled = n_unlabeled-len(list_of_removed)
        if(len(list_of_removed)!=0): 
            new_true_labels = np.delete(true_labels_copy, list_of_removed)
        new_labels = np.zeros(n_labeled+new_n_unlabeled)
        new_labels[:n_labeled] = true_labels_copy[:n_labeled]
        
    labels = np.zeros(n_labeled+n_unlabeled)
    labels[:n_labeled] = true_labels[:n_labeled]

    steps = np.linspace(15, n_labeled, 10)
    #steps = [20]
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
        labels = labels_save[n_labeled_save - n_labeled : n_unlabeled+n_labeled_save]
        adj_matrix = adj_matrix_save[n_labeled_save - n_labeled : n_unlabeled+n_labeled_save, n_labeled_save - n_labeled : n_unlabeled+n_labeled_save]
        
        if(is_square(adj_matrix) ==False):
            print("truuuuuuuuuuuueeeeeeeeeeeeeeeee")
        
        
        nearest_neighbor_labels = nearest_neighbor_from_labeled(labels,adj_matrix, n_labeled, n_unlabeled)  
       # nearest_neighbor_labels1 = init_nearest_labels(n_labeled, n_unlabeled, adj_matrix, labels)
        errors_indexes_nn = np.where(np.not_equal(nearest_neighbor_labels,true_labels[n_labeled_save -n_labeled: n_unlabeled+n_labeled_save]))
        sum_of_errors_nn = np.size(errors_indexes_nn,1)
        
       # errors_indexes_nn1 = np.where(np.not_equal(nearest_neighbor_labels1,true_labels[n_labeled_save -n_labeled: n_unlabeled+n_labeled_save]))
       # sum_of_errors_nn1 = np.size(errors_indexes_nn1,1)
        
            
        nn_errors.append(sum_of_errors_nn)
        
        #print("nn: errors_indexes_nearest_neighbor:",errors_indexes_nn)
        #print("nn: errors_sum_nearest_neighbor:",sum_of_errors_nn)
        #print("nn1: errors_indexes_nearest_neighbor:",errors_indexes_nn1)
        #print("nn1: errors_sum_nearest_neighbor:",sum_of_errors_nn1)


        k=10   
        
        knn_labels,new_algo_labels = get_labels_median_k_nn(labels,k,adj_matrix, n_labeled, n_unlabeled)
        errors_indexes_knn = np.where(np.not_equal(knn_labels, true_labels[n_labeled_save -n_labeled: n_unlabeled+n_labeled_save]))
        sum_of_errors_knn = np.size(errors_indexes_knn,1)
        knn_errors.append(sum_of_errors_knn)
        #print("knn: errors_indexes_nearest_neighbor:",errors_indexes_knn)
        #print("knn: errors_sum_nearest_neighbor:",sum_of_errors_knn)
        

        errors_indexes_na = np.where(np.not_equal(new_algo_labels, true_labels[n_labeled_save -n_labeled: n_unlabeled+n_labeled_save]))
        sum_of_errors_na = np.size(errors_indexes_na,1)
        na_errors.append(sum_of_errors_na)
        #print("na: errors_indexes_nearest_neighbor:",errors_indexes_na)
        #print("na: errors_sum_nearest_neighbor:",sum_of_errors_na)
        
        n_outliers = 4
        n_out_labels = get_labels_n_outliers_from_k_nn(labels,n_outliers,k,adj_matrix, n_labeled, n_unlabeled)
        errors_indexes_no = np.where(np.not_equal(n_out_labels, true_labels[n_labeled_save-n_labeled: n_unlabeled+n_labeled_save]))
        sum_of_errors_no = np.size(errors_indexes_no,1)
        no_errors.append(sum_of_errors_no)
        #print("no: errors_indexes_nearest_neighbor:",errors_indexes_no)
        #print("no: errors_sum_nearest_neighbor:",sum_of_errors_no)

        #print("na: errors_indexes_nearest_neighbor:",errors_indexes_na)
        #print("na: errors_sum_nearest_neighbor:",sum_of_errors_na)
        
        
        #new_algo_labels_2 = get_labels_corr_threshold(labels,k,adj_matrix, n_labeled, n_unlabeled)
        #errors_indexes_na_2 = np.where(np.not_equal(new_algo_labels_2, true_labels[n_labeled_save -n_labeled: n_unlabeled+n_labeled_save]))
        #sum_of_errors_na_2 = np.size(errors_indexes_na_2,1)
        #na_errors_2.append(sum_of_errors_na_2)
        #print("na_2: errors_indexes_nearest_neighbor_2:",errors_indexes_na_2)
        #print("na_2: errors_sum_nearest_neighbor_2:",sum_of_errors_na_2)
        
        if j==2 and n_labeled ==n_labeled_save:
            nearest_neighbor_labels[list_of_removed]=-1
            knn_labels[list_of_removed]=-1
            new_algo_labels[list_of_removed]=-1
            n_out_labels[list_of_removed] =-1
        
        min_errors = [sum_of_errors_nn,sum_of_errors_knn,sum_of_errors_na,sum_of_errors_no]
        
        
        print("gaussian_num:",gaussian_num)
        print("n_labeled:",n_labeled)
        print(np.min(min_errors))
        print(np.argmin(min_errors))
        
        



    #labels1, list_index_of_erors1 = run_process(n_labeled, n_unlabeled, true_labels, adj_matrix, 0)

    n_labeled = n_labeled_save
    labels = labels_save
    #visual_error_2(n_labeled, n_unlabeled, adj_matrix, new_algo_labels,true_labels,imgs)
    visual_error_2(n_labeled, n_unlabeled, adj_matrix, nearest_neighbor_labels, true_labels_copy, imgs)

    plt.plot(steps,nn_errors,'-o', label="Nearest Neighbor")
    plt.plot(steps,knn_errors,'-o', label="K Nearest Neighbor")
    plt.plot(steps,na_errors,'->', label="new_algo")
    plt.plot(steps,no_errors,'-^', label="4 outliers in NN as outlier")
    #plt.plot(steps,na_errors_2,'-o', label="new_algo_2")




    plt.legend()
    if(wrong_removed):
        str_remove = " wrong_removed"
    else:
        str_remove = " and "+str(len(list_of_removed))+" outliers removed"
    if j==0 or j==2:
        str_remove = " all images"
        
    title = "errors: " + str(np.min(min_errors)) + str_remove
    plt.xlabel("num of labeled data")
    plt.ylabel("num of errors")
    plt.title(title)
    plt.show()
      
    print("lololol")
    #calc_sum_corr(adj_matrix, true_labels)
    #visual_error_2(n_labeled, n_unlabeled, adj_matrix, nearest_neighbor_labels, true_labels, imgs)
    adj_matrix_analysis(adj_matrix, true_labels)
#im = outliers_imgs[i]
#fig, ax = plt.subplots(1,2,figsize=(15,15))
#ax[0].imshow(im, cmap='gray')
#ax[0].set_title('original: outlier', fontsize = f_size)
#ax[1].imshow(ndimage.gaussian_filter(im, 20), cmap = 'gray')
#ax[1].set_title('gaussian filter 20', fontsize = f_size);
#plt.show()
#i = i+1
