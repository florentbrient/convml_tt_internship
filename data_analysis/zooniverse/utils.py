import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import sklearn.decomposition
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
#from ...data.dataset import ImageSingletDataset, TileType
from convml_tt.data.dataset import TileType, ImageSingletDataset
import pickle
from sklearn.metrics import silhouette_score
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection


import pytorch_lightning as pl
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import torch
from sklearn.decomposition import PCA
from PIL import Image
import os, shutil
from matplotlib.patches import Rectangle as rectan
from random import seed
from random import randint
from sklearn.metrics import confusion_matrix
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestCentroid
from tqdm.notebook import tqdm
TILE_FILENAME_FORMAT = "{triplet_id:05d}_{tile_type}.png"
TEST_SIZE = 0.3
RANDOM_STATE = 1024
COLORS = ['b', 'g', 'r', 'm'] # Color of each class
#DATASET_DIR = "../../NC/zooniverse/"
print(cv2.__version__)

pca = PCA(n_components=3, svd_solver='arpack')

def rle2mask(mask_rle, shape=(2100, 1400)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (width,height) of array to return 
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T

def extract_non_black(img):
    # Import your picture
    input_picture = img.copy()
    #input_picture = cv2.cvtColor(input_picture, cv2.COLOR_BGR2RGB) 
    # Color it in gray
    gray = cv2.cvtColor(input_picture, cv2.COLOR_BGR2GRAY)

    # Create our mask by selecting the non-zero values of the picture
    ret, mask = cv2.threshold(gray,0, 255,cv2.THRESH_BINARY)

    # Select the contour
    cont, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # if your mask is incurved or if you want better results, 
    # you may want to use cv2.CHAIN_APPROX_NONE instead of cv2.CHAIN_APPROX_SIMPLE, 
    # but the rectangle search will be longer
    
    cv2.drawContours(input_picture, cont, -1, (0,255,255), 2)
    """
    plt.figure()
    plt.imshow(input_picture)
    plt.show()
    """
    # Find contour and sort by contour area
    gray = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)[1]
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    # Find bounding box and extract ROI
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        ROI = img[y:y+h, x:x+w]
        #return ROI
        break
    mina = min(ROI.shape[0], ROI.shape[1])
    return ROI[0:mina, 0:mina]
    plt.figure()
    plt.imshow(ROI[0:mina, 0:mina])
    plt.show()
    

def extract_contour(img, cnts, shown=False, direct=True):
    '''
        Extract from contours images
    '''
    #cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    rois = []
    # Find bounding box and extract ROI
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        ROI = img[y:y+h, x:x+w]
        if direct:
            return ROI
        rois.append(ROI)
    if not shown:
        plt.figure()
        plt.imshow(ROI)
    return rois



def predict_domain(image):
    '''
        for a specific image of zooniverse predict each tile.
    '''
    deb_path = "../../tmp/train/"
    i = 0
    for x in range(0, image.shape[0], 256):
            for y in range(0, image.shape[1], 256):
                out_name =  TILE_FILENAME_FORMAT.format(triplet_id=i,tile_type='anchor')
                out_name = deb_path+out_name
                print("out_name : "+str(out_name)+" shape : "+str(image[x:x+256,y:y+256].shape))
                
                if image[x:x+256,y:y+256].shape != (256, 256, 3):
                    continue
                to_save = cv2.cvtColor(image[x:x+256,y:y+256], cv2.COLOR_BGR2RGB)
                cv2.imwrite(out_name, to_save)
                i += 1
    print("Number of images : "+str(i))
    dataset_path =  'tmp/'
    tile_dataset = ImageSingletDataset(data_dir=dataset_path, tile_type=TileType.ANCHOR)
    da_embeddings = get_embeddings(tile_dataset=tile_dataset, model=model)
    compressed = pca.fit_transform(da_embeddings)
    print("Embedding shape : "+str(da_embeddings.shape))
    print("Compressed PCA shape : "+str(compressed.shape))
    plt.imshow(compressed.T)
    return da_embeddings
    

def visualize_boxes_cluster(df, sample):
    #tmp_dataset_path =  '../../../NC/tmp'
    #print(tmp_dataset_path)
    tmp_dataset_path = '../../NC/division'
    #print(tmp_dataset_path)
    for filename in os.listdir(tmp_dataset_path+"/train/"):
        file_path = os.path.join(tmp_dataset_path+"/train/", filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
    coordinates = []
    t = 0
    
    fig, ax = plt.subplots(figsize=(15, 10))
    img_path = os.path.join(DATASET_DIR, 'train_images', sample[0])
    img = cv2.imread(img_path, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Get annotations
    labels = df[df['Image_Label'].str.contains(sample[0])]['EncodedPixels']
    patches = []
    for idx, rle in enumerate(labels.values):
        if rle is not np.nan:
            mask = rle2mask(rle)
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            cnts = sorted(contours, key=cv2.contourArea, reverse=True)
            rois = []
            # Find bounding box and extract ROI
            xvs,yvs,wvs,hvs = [], [], [], []
            for c in cnts:
                xv,yv,wv,hv = cv2.boundingRect(c)
                xvs.append(xv)
                yvs.append(yv)
                wvs.append(wv)
                hvs.append(hv)
                #break
            #print("Contours : "+str(contours))
            rgbImages = extract_contour(img, contours, direct=False)
            da = 0
            for rgbImg in rgbImages:
                rgbImg = cv2.cvtColor(rgbImg, cv2.COLOR_BGR2RGB)
                for x in range(0, rgbImg.shape[0], 256):
                    for y in range(0, rgbImg.shape[1], 256):
                        if rgbImg[x:x+256,y:y+256].shape != (256, 256, 3):# or  [0,0,0] in rgbImg[x:x+256,y:y+256]:
                            continue
                        
                        tm = t
                        coordinates.append([xvs[da]+x,yvs[da]+y, wvs[da], hvs[da]])
                        out_name = TILE_FILENAME_FORMAT.format(triplet_id=tm,tile_type='anchor')
                        out_name = '../../NC/tmp/train/'+out_name
                        cv2.imwrite(out_name,rgbImg[x:x+256,y:y+256])
                        t += 1
                da += 1
            for contour in contours:
                poly_patch = Polygon(contour.reshape(-1, 2), closed=True, linewidth=2, edgecolor=COLORS[idx], facecolor=COLORS[idx], fill=True)
                patches.append(poly_patch)
    p = PatchCollection(patches, match_original=True, cmap=matplotlib.cm.jet, alpha=0.3)
    ax.imshow(img/255)
    ax.set_title('{} - ({})'.format(sample[0], ', '.join(sample[1].astype(np.str))))
    ax.add_collection(p)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    plt.show()
    
    tmp_tile_dataset = ImageSingletDataset(data_dir=tmp_dataset_path, tile_type=TileType.ANCHOR)
    tmp_da_embeddings = get_embeddings(tile_dataset=tmp_tile_dataset, model=model)
    tmp_clusters = convml_tt.interpretation.plots.dendrogram(tmp_da_embeddings, n_samples=10, n_clusters_max=12, return_clusters=True)

    fig, ax = plt.subplots(figsize=(15, 10))
    colors = ['red', 'azure', 'forestgreen', 'mediumblue', 'goldenrod', 'olivedrab', 'chocolate', 'darkseagreen', 'steelblue', 'dodgerblue', 'crimson', 'darkorange']
    # Display the image
    #rgbImg = cv2.cvtColor(rgbImg, cv2.COLOR_BGR2RGB)
    ax.imshow(img/255)

    # Create a Rectangle patch
    cnt = 0
    for coordinate in coordinates:
        rect = rectan((coordinate[0], coordinate[1]), coordinate[2], coordinate[3], linewidth=1, facecolor=colors[tmp_clusters[1][cnt]]  , edgecolor=colors[tmp_clusters[1][cnt]], fill=True,  alpha=0.3)
        #poly_patch = Polygon(contour.reshape(-1, 2), closed=True, linewidth=2, edgecolor=COLORS[idx], facecolor=COLORS[idx], fill=True)
        #patches.append(poly_patch)
        cnt += 1
        # Add the patch to the Axes
        ax.add_patch(rect)
    ax.set_title("Per tile cluster prediction")
    ax.set_ylabel("KM")
    ax.set_xlabel("KM")
    plt.show()


def visualize_domain(df, sample):
    tmp_dataset_path =  '../../../NC/tmp'
    for filename in os.listdir(tmp_dataset_path+"/train/"):
        file_path = os.path.join(tmp_dataset_path+"/train/", filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
    coordinates = []
    t = 0
    
    fig, ax = plt.subplots(figsize=(15, 10))
    img_path = os.path.join(DATASET_DIR, 'train_images', sample[0])
    img = cv2.imread(img_path, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Get annotations
    labels = df[df['Image_Label'].str.contains(sample[0])]['EncodedPixels']

    patches = []
    
    for x in range(0, img.shape[0], 256):
        for y in range(0, img.shape[1], 256):
            if img[x:x+256,y:y+256].shape != (256, 256, 3):# or  [0,0,0] in rgbImg[x:x+256,y:y+256]:
                continue
            tm = t
            coordinates.append([x, y, 256, 256])
            out_name = TILE_FILENAME_FORMAT.format(triplet_id=tm,tile_type='anchor')
            out_name = '../../../NC/tmp/train/'+out_name
            cv2.imwrite(out_name, img[x:x+256,y:y+256])
            t += 1
    for idx, rle in enumerate(labels.values):
        if rle is not np.nan:
            mask = rle2mask(rle)
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                poly_patch = Polygon(contour.reshape(-1, 2), closed=True, linewidth=2, edgecolor=COLORS[idx], facecolor=COLORS[idx], fill=True)
                patches.append(poly_patch)
    p = PatchCollection(patches, match_original=True, cmap=matplotlib.cm.jet, alpha=0.3)
    ax.imshow(img/255)
    ax.set_title('{} - ({})'.format(sample[0], ', '.join(sample[1].astype(np.str))))
    ax.add_collection(p)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    plt.show()
    
    tmp_tile_dataset = ImageSingletDataset(data_dir=tmp_dataset_path, tile_type=TileType.ANCHOR)
    tmp_da_embeddings = get_embeddings(tile_dataset=tmp_tile_dataset, model=model)
    tmp_clusters = convml_tt.interpretation.plots.dendrogram(tmp_da_embeddings, n_samples=10, n_clusters_max=12, return_clusters=True)

    fig, ax = plt.subplots(figsize=(15, 10))
    colors = ['red', 'azure', 'forestgreen', 'mediumblue', 'goldenrod', 'olivedrab', 'chocolate', 'darkseagreen', 'steelblue', 'dodgerblue', 'crimson', 'darkorange']
    # Display the image
    ax.imshow(img/255)

    # Create a Rectangle patch
    cnt = 0
    for coordinate in coordinates:
        rect = rectan((coordinate[1], coordinate[0]), coordinate[2], coordinate[3], linewidth=1, facecolor=colors[tmp_clusters[1][cnt]]  , edgecolor=colors[tmp_clusters[1][cnt]], fill=True,  alpha=0.3)
        cnt += 1
        # Add the patch to the Axes
        ax.add_patch(rect)
    ax.set_title("Per tile cluster prediction")
    ax.set_ylabel("KM")
    ax.set_xlabel("KM")
    plt.show()






def kmeans(
    da_embeddings,
    ax=None,
    visualize=False,
    model_path=None,
    n = 12,
    method=None,
    save=False
):
    """
    K-Means clustering.
    """

    tile_dataset = ImageSingletDataset(
        data_dir=da_embeddings.data_dir,
        tile_type=da_embeddings.tile_type,
        stage=da_embeddings.stage,
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 3))
    else:
        fig = ax.figure

    if model_path is None:
        if method == "optimize":
            sse = []
            silhouette_coefficients = []
            best_silouhette = -999
            best_silouhette_k = -1
            best_clusters = []
            r = 0
            for k in range(2, 30):
                kmeans = KMeans(n_clusters=k)
                clusters = kmeans.fit_predict(da_embeddings)
                sse.append(kmeans.inertia_)
                #centers = clusterer.cluster_centers_
                score = silhouette_score(da_embeddings, clusters)
                silhouette_coefficients.append(score)
                if score>best_silouhette:
                    best_silouhette = score
                    best_silouhette_k = kmeans
                    best_clusters = clusters
            kmeans = best_silouhette_k
            clusters = best_clusters
            if visualize:
                fig = plt.gcf()
                fig.set_size_inches(20, 20)
                plt.style.use("fivethirtyeight")
                plt.plot(range(2, 30), sse)
                plt.xticks(range(1, 30))
                plt.xlabel("Number of Clusters")
                plt.ylabel("SSE")
                plt.show()

                fig = plt.gcf()
                fig.set_size_inches(20, 20)
                plt.style.use("fivethirtyeight")
                plt.plot(range(2, 30), silhouette_coefficients)
                plt.xticks(range(2, 30))
                plt.xlabel("Number of Clusters")
                plt.ylabel("Silhouette Coefficient")
                plt.show()
        else:
            kmeans = KMeans(
                    init="random",
                    n_clusters=n,
                    n_init=10,
                    max_iter=300,
                    random_state=42 
                )
            clusters = kmeans.fit_predict(da_embeddings)
    else:
        kmeans = pickle.load(open("kmeans_model.pkl", "rb"))
        clusters = kmeans.predict(da_embeddings)
        return clusters, kmeans
    if save:
        pickle.dump(kmeans, open("kmeans_model.pkl", "wb"))    

    return clusters, kmeans
