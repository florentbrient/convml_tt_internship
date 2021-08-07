{"64": 0.15341751833499875, "128": 0.1145085735801374, "256": 0.08755108343078824, "512": 0.045215654188946186}

64 scale have NMI = 0.00590539424538167

128 scale have NMI = 0.024199582352905088

256 scale have NMI = 0.0768392537668485

512 scale have NMI = 0.16038507354608583

612 scale have NMI = 0.1671278207930043

850 scale have NMI = 0.42465641591369907

# ORIGINAL 

import os
divisions = [64, 128, 256, 512, 612, 850, 1024]
for div in divisions:
    dataset_path =  DATASETS_path+'MULTI_SCALE/'+str(div)+'/'
    print("DATASET : "+str(dataset_path))
    print(os.listdir(dataset_path))
    tile_dataset = ImageSingletDataset(data_dir=dataset_path, tile_type=TileType.ANCHOR)
    da_embeddings = get_embeddings(tile_dataset=tile_dataset, model=model)
    clusters = convml_tt.interpretation.plots.dendrogram(da_embeddings, n_samples=10, n_clusters_max=12, return_clusters=True)#, label_clusters=True)
    with open(DATASETS_path+"MULTI_SCALE/"+str(div)+"/labels.pickle", "rb") as handle:
        labs = pickle.load(handle)
    
    print(len(labs))
    print(len(clusters[1]))
    clclc = list(map(int, clusters[1]))
    cfs_mat = confusion_matrix(labs, clclc)

    associations = ['Fish', 'Flower', 'Gravel', 'Sugar']
    fig, ax = plt.subplots(figsize=(20,10)) 
    plt.title("Confusion Matrix Clusters MODIS model SCALE "+str(div))
    sns.set(font_scale=1.8)
    sns.heatmap(cfs_mat[:4,:], annot=True, fmt='', ax=ax, linewidths=.9, yticklabels=associations)
    fig, ax = plt.subplots(figsize=(20,10))
    print(cfs_mat[:4,:].shape)
    print(   np.sum(  cfs_mat[:4,:], axis=1  ).shape  )
    print(   (  cfs_mat[:4,:].T/np.sum(cfs_mat[:4,:], axis=1)    ).shape  )
    plt.title("Confusion Matrix Clusters MODIS model %  SCALE "+str(div))
    sns.heatmap( (cfs_mat[:4,:].T/np.sum(cfs_mat[:4,:], axis=1)).T, annot=True, 
                fmt='.2%', cmap='Blues' , yticklabels=associations)
    print(len(da_embeddings))
    print(len(clusters[1]))
    nmi = normalized_mutual_info_score(labs, clusters[1])
    print(str(div)+" scale have NMI = "+str(nmi))
    nmis[str(div)] = nmi
    pca_2d = PCA(n_components=2, svd_solver='arpack')
    principalComponents = pca_2d.fit_transform(da_embeddings)
    print(principalComponents.shape)
    print(principalComponents[:, 0].shape)
    print(principalComponents[:, 0])
    print("\n\n\n")
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split( principalComponents, clusters[1], test_size=0.01, random_state=42)

    fig, ax = plt.subplots(figsize=(15,15))

    ax.scatter(principalComponents[:, 0], principalComponents[:, 1], c=clusters[1], s=1)
    plt.title('Scatter plot of images embedding depending PC SCALE'+str(div))
    ax.set_xlabel('PC 1')
    ax.set_ylabel('PC 2')
    
    for cluster_number in range(0, 12):
        indices = np.where(clusters[1]==cluster_number)[0]
        idx = np.random.choice(indices)
        tm = TILE_FILENAME_FORMAT.format(triplet_id=idx,tile_type='anchor')
        filename = DATASETS_path+"MULTI_SCALE/"+str(div)+"/train/"+tm
        print("filename : "+str(filename))
        arr_img = plt.imread(filename, format='png')    
        imagebox = OffsetImage(arr_img, zoom=0.2)
        xy = (principalComponents[idx, 0], principalComponents[idx, 1])
        
        ab = AnnotationBbox(imagebox, xy,
            xybox=(30, -30),
            xycoords='data',
            boxcoords="offset points")
        ax.add_artist(ab)


    plt.show()

    print(X_test[:, 0].shape)
    print(X_test[:, 0])
    print("\n\n\n")
    fig, ax = plt.subplots(figsize=(15,15))
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, s=20)
    plt.title('Scatter plot of a sample of images embedding depending PC SCALE '+str(div))
    ax.set_xlabel('PC 1')
    ax.set_ylabel('PC 2')
    plt.show()