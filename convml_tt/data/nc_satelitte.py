import enum
from pathlib import Path
from typing import Union
import netCDF4 as nc
import numpy as np
import luigi
import pickle
from PIL import Image 
import PIL
import xarray
import cv2
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import random 
from math import sqrt, cos, sin

TILE_FILENAME_FORMAT = "{triplet_id:05d}_{tile_type}.png"
def NormalizeData(arr):
    '''
        Normalize array between [0, 255]
    '''
    return ((arr - arr.min()) * (1/(arr.max() - arr.min()) * 255)).astype('uint8')

def norm(x):
    '''
        Normalize array between [0, 1]
    '''
    return (x-x.min())/(x.max()-x.min())

def contrast_correction(color, contrast):
    """
    Modify the contrast of an R, G, or B color channel
    See: #www.dfstudios.co.uk/articles/programming/image-programming-algorithms/image-processing-algorithms-part-5-contrast-adjustment/
    Input:
        C - contrast level
    """
    F = (259*(contrast + 255))/(255.*259-contrast)
    COLOR = F*(color-.5)+.5
    COLOR = np.minimum(COLOR, 1)
    COLOR = np.maximum(COLOR, 0)
    return COLOR

def is_convective(file):
    ds = nc.Dataset(file, 'r')
    ir = ds.variables['temp_11_0um_nom'][:]
    if np.percentile(ir, 25)>285:
        return True
    return False

def rebin(a, shape):
    sh = shape[0],a.shape[0]//shape[0],shape[1],a.shape[1]//shape[1]
    return a.reshape(sh).mean(-1).mean(1)

def transform_nc(file, plot=False):
    # Reading in GOES-16 NetCDF
    g16nc = nc.Dataset(file, 'r')
    radiance = g16nc.variables['refl_0_65um_nom'][:]

    if plot:
        # Initial Radiance Plot
        fig = plt.figure(figsize=(6,6),dpi=200)
        im = plt.imshow(radiance, cmap='Greys_r')
        cb = fig.colorbar(im, orientation='horizontal')
        cb.set_ticks([1, 100, 200, 300, 400, 500, 600])
        cb.set_label('Radiance (W m-2 sr-1 um-1)')
        plt.show()

    # Radiance to Reflectance
    Esun_Ch_01 = 726.721072
    Esun_Ch_02 = 663.274497
    Esun_Ch_03 = 441.868715
    d2 = 0.3

    # Apply the formula to convert radiance to reflectance
    ref = (radiance * np.pi * d2) / Esun_Ch_02
    # Make sure all data is in the valid data range
    ref = np.maximum(ref, 0.0)
    ref = np.minimum(ref, 1.0)

    if plot:
        # Plot reflectance
        fig = plt.figure(figsize=(6,6),dpi=200)
        im = plt.imshow(ref, vmin=0.0, vmax=1.0, cmap='Greys_r')
        cb = fig.colorbar(im, orientation='horizontal')
        cb.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        cb.set_label('Reflectance')
        plt.show()

    # Gamma correction
    # Apply the formula to adjust reflectance gamma
    ref_gamma = np.sqrt(ref)

    if plot:
        # Plot gamma adjusted reflectance
        fig = plt.figure(figsize=(6,6),dpi=200)
        im = plt.imshow(ref_gamma, vmin=0.0, vmax=1.0, cmap='Greys_r')
        cb = fig.colorbar(im, orientation='horizontal')
        cb.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        cb.set_label('Reflectance')
        plt.show()

    # Psuedo-True Color Image
    # Load Channel 1 - Blue Visible
    
    radiance_1 = g16nc.variables['refl_0_47um_nom'][:]
    
    ref_1 = (radiance_1 * np.pi * d2) / Esun_Ch_01
    # Make sure all data is in the valid data range
    ref_1 = np.maximum(ref_1, 0.0)
    ref_1 = np.minimum(ref_1, 1.0)
    ref_gamma_1 = np.sqrt(ref_1)

    if plot:
        # Plot gamma adjusted reflectance channel 1
        fig = plt.figure(figsize=(6,6),dpi=200)
        im = plt.imshow(ref_gamma_1, vmin=0.0, vmax=1.0, cmap='Greys_r')
        cb = fig.colorbar(im, orientation='horizontal')
        cb.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        cb.set_label('Ch01 - Reflectance')
        plt.show()

    # Load Channel 3 - Veggie Near IR
    g16nc = nc.Dataset(file, 'r')
    radiance_3 = g16nc.variables['refl_0_86um_nom'][:]
    g16nc.close()
    g16nc = None
    ref_3 = (radiance_3 * np.pi * d2) / Esun_Ch_03
    # Make sure all data is in the valid data range
    ref_3 = np.maximum(ref_3, 0.0)
    ref_3 = np.minimum(ref_3, 1.0)
    ref_gamma_3 = np.sqrt(ref_3)

    if plot:
        # Plot gamma adjusted reflectance channel 3
        fig = plt.figure(figsize=(6,6),dpi=200)
        im = plt.imshow(ref_gamma_3, vmin=0.0, vmax=1.0, cmap='Greys_r')
        cb = fig.colorbar(im, orientation='horizontal')
        cb.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        cb.set_label('Ch03 - Reflectance')
        plt.show()

    ref_gamma_2 = rebin(ref_gamma, list(ref_gamma_3.shape))
    geocolor = np.stack([ref_gamma_2, ref_gamma_3, ref_gamma_1], axis=2)

    if plot:
        fig = plt.figure(figsize=(6,6),dpi=200)
        im = plt.imshow(geocolor)
        plt.title('GeoColor - Red - Veggie - Blue')
        plt.show()
    
    # Derived from Planet Labs data, CC > 0.9
    ref_gamma_true_green = 0.48358168 * ref_gamma_2 + 0.45706946 * ref_gamma_1 + 0.06038137 * ref_gamma_3
    if plot:
        truecolor = np.stack([ref_gamma_2, ref_gamma_true_green, ref_gamma_1], axis=2)
        fig = plt.figure(figsize=(6,6),dpi=200)
        im = plt.imshow(truecolor)
        plt.title('TrueColor - Red - Psuedo-Green - Blue')
        plt.show()
        # We'll use the `CMI_C02` variable as a 'hook' to get the CF metadata.
    
    gamma = 0.25
    B, R, G = ref_1, ref, ref_3
    R = np.power(R, gamma)
    G = np.power(G, gamma)
    B = np.power(B, gamma)
    # print '\n   Gamma correction: %s' % gamma

    # Calculate the "True" Green
    G_true = 0.48358168 * R + 0.45706946 * B + 0.06038137 * G
    G_true = np.maximum(G_true, 0)
    G_true = np.minimum(G_true, 1)

    # Modify the RGB color contrast:
    contrast = 300
    arr_img = np.dstack([R, G_true, B])
    arr_img = contrast_correction(arr_img, contrast)
    if plot:
        fig = plt.figure(figsize=(6,6),dpi=200)
        im = plt.imshow(arr_img)
        plt.show()
    return arr_img



class Extract_from_nc_images(luigi.Task): 
    def run(self,  padding=10, selective_hour=True):
        content = self.input().open('r').read().splitlines()
        #print("Content : \n"+str(content))
        all_files_channels = []
        t = 0
        v = 0
        lower_val = np.array([190,0,0]) 
        upper_val = np.array([200,255,255]) 
        not_red = 0
        num_images = 0
        dic_images = {}
        DATASETS_path = "~/tmp/"
        train_saving_path = DATASETS_path+'train/'
        study_saving_path = DATASETS_path+'study/'
        saving_path = train_saving_path
        PI = 3.1456
        t = 0
        index_img = 0


        for i in tqdm(content):
            #print("file n°"+str(t)+" images generated : "+str(t))
            if index_img < 0.81*len(content):
                    saving_path = train_saving_path
            else:
                saving_path = study_saving_path
            
            ds = nc.Dataset(i)
            if selective_hour and ( float(ds.__dict__["START_TIME"])>13.20 or ds.__dict__["START_TIME"]<12.90 ):
                continue
            if "BARBADOS" not in ds.__dict__['SUBSET_NAME']:
                continue
            
            #RGB_contrast = np.dstack([R, G, B])
            RGB_contrast = transform_nc(i)
            RGB_contrast = NormalizeData(RGB_contrast)
            if not is_convective(i):
                continue
            #plt.imshow(RGB_contrast)
            #plt.show()
            HSV = cv2.cvtColor(RGB_contrast, cv2.COLOR_RGB2HSV)
            for x in range(0,RGB_contrast.shape[0], 256):
                for y in range(0, RGB_contrast.shape[1], 256):
                    # Threshold the HSV image - any green color will show up as white
                    mask = cv2.inRange(HSV[x:x+256,y:y+256], (100, 0, 0), (101, 0, 0))

                    # if there are any white pixels on mask, sum will be > 0
                    
                    img = HSV[x:x+256,y:y+256] #[:,:,0]
                    #hasGreen = np.sum(mask)
                    hasGreen = False
                    #print("AAAH ? "+str(np.unique(RGB_contrast[x:x+256,y:y+256]).shape))
                    #hasGreen = np.logical_or(RGB_contrast[x:x+256,y:y+256] == [18,46,63] , RGB_contrast[x:x+256,y:y+256] == [30,55,67]).any()
                    """
                    if hasGreen:
                        print("Found continent !")
                    """
                    if RGB_contrast[x:x+256,y:y+256].shape != (256, 256, 3) or hasGreen or np.unique(RGB_contrast[x:x+256,y:y+256]).shape==(2,):
                        """
                        out_name = TILE_FILENAME_FORMAT.format(triplet_id=v,tile_type='anchor')
                        out_name = 'tmp/not_taken/'+out_name
                        im = Image.fromarray(RGB_contrast[x:x+256,y:y+256])
                        im.save(out_name)
                        v += 1
                        """
                        pass
                    else:
                        start = time.time()
                        time.clock()    
                        found = False # Variable meaning that we didn't find suiting neighobr tile
                        best_neighbor = 0
                        best_neighbor_img = []
                        while not found: # Getting a neighbor image
                            R = 256/2
                            theta = random.random() * 2 * PI

                            
                            x_shift, y_shift = int(R * cos(theta)), int(R * sin(theta))
                            neighobr_img = RGB_contrast[x+x_shift:x+x_shift+256,y+y_shift:y+y_shift+256]
                            elapsed = time.time() - start
                            if len(np.unique(neighobr_img.reshape(-1, neighobr_img.shape[2]), axis=0))>best_neighbor and neighobr_img.shape == (256, 256, 3):
                                best_neighbor = len(np.unique(neighobr_img.reshape(-1, neighobr_img.shape[2]), axis=0))
                                best_neighbor_img = neighobr_img

                            if (not (neighobr_img.shape != (256, 256, 3) or len(np.unique(neighobr_img.reshape(-1, neighobr_img.shape[2]), axis=0))<30)) or elapsed>30:
                                found = True
                        if [0, 0, 0] in best_neighbor_img:
                            continue
                        neighobr_img = best_neighbor_img
                        found = False # distant

                        while not found:
                            
                            rand_index = i 
                            #RGB_contrast = np.dstack([R, G, B])
                            while rand_index == i:
                                rand_index = random.randint(0, len(content)-1)
                            
                            ds_dist = nc.Dataset(content[rand_index])
                            if selective_hour and ( float(ds_dist.__dict__["START_TIME"])>13.20 or ds_dist.__dict__["START_TIME"]<12.90 ):
                                continue
                            if "BARBADOS" not in ds_dist.__dict__['SUBSET_NAME']:
                                continue

                            random_img = transform_nc(content[rand_index])
                            #print("RANDOM IMG SHAPE : "+str(random_img.shape))
                            distant_img = NormalizeData(random_img)
                            #print("DISTANT IMG SHAPE : "+str(random_img.shape))
                            #distant_img = cv2.cvtColor(distant_img, cv2.COLOR_BGR2RGB)
                            x_distant = random.randint(0, distant_img.shape[0]-257)
                            y_distant = random.randint(0, distant_img.shape[1]-257)
                            distant_img_shaped = distant_img[x_distant:x_distant+256,y_distant:y_distant+256]
                            if not (distant_img_shaped.shape != (256, 256, 3) or [0,0,0] in distant_img_shaped):
                                found = True
                            
                        tm = t
                        
                        # anchor
                        out_name = TILE_FILENAME_FORMAT.format(triplet_id=tm,tile_type='anchor')
                        out_name = saving_path + out_name
                        #print("shape : "+str(RGB_contrast[x:x+256,y:y+256].shape))
                        #print("Save anchor : "+str(out_name))
                        cv2.imwrite(out_name, RGB_contrast[x:x+256,y:y+256])
                        

                        # neighbor
                        out_name_neighbor = TILE_FILENAME_FORMAT.format(triplet_id=tm,tile_type='neighbor')
                        out_name_neighbor = saving_path + out_name_neighbor
                        #print("shape : "+str(neighobr_img.shape))
                        #print("Save neighbor : "+str(out_name_neighbor))
                        #print("unique : "+str(np.unique(neighobr_img)))
                        cv2.imwrite(out_name_neighbor, neighobr_img)
                        

                        # distant
                        out_name_distant = TILE_FILENAME_FORMAT.format(triplet_id=tm,tile_type='distant')
                        out_name_distant = saving_path + out_name_distant
                        #print("shape : "+str(distant_img_shaped.shape))
                        #print("Save distant : "+str(out_name_distant))
                        #print("unique : "+str(np.unique(distant_img_shaped)))
                        cv2.imwrite(out_name_distant, distant_img_shaped)
                        

                        #print("\n\n")
                        
                        dic_images[out_name] = {}
                        dic_images[out_name]['START_TIME'] = ds.__dict__['START_TIME']
                        dic_images[out_name]['END_TIME'] = ds.__dict__['END_TIME']
                        dic_images[out_name]['date_created'] = ds.__dict__['date_created']
                        dic_images[out_name]['START_DAY'] = ds.__dict__['START_DAY']
                        dic_images[out_name]['END_DAY'] = ds.__dict__['END_DAY']
                        dic_images[out_name]['LAT_SOUTH_SUBSET'] = ds.__dict__['LAT_SOUTH_SUBSET']
                        dic_images[out_name]['LAT_NORTH_SUBSET'] = ds.__dict__['LAT_NORTH_SUBSET']
                        dic_images[out_name]['LON_WEST_SUBSET'] = ds.__dict__['LON_WEST_SUBSET']
                        dic_images[out_name]['LON_EAST_SUBSET'] = ds.__dict__['LON_EAST_SUBSET']
                        dic_images[out_name]["CROP"] = [x, x+256, y, y+256]

                        t += 1

        with self.output().open("w") as f:
            f.write(str(dic_images))
    def output(self):
        return luigi.LocalTarget('tmp/channels.pickle')

    def requires(self):
        return ExtractNcFiles() 

class ExtractNcFiles(luigi.Task):
    path = luigi.Parameter()
    def run(self):
        files = list()
        for (dirpath, dirnames, filenames) in os.walk(self.path):
            files += [os.path.join(dirpath, file) for file in filenames]
        with self.output().open("w") as f:
            f.write('\n'.join(files))
        

    def output(self):
        return luigi.LocalTarget("tmp/files.txt")


class BuildImages(luigi.Task):
    def run(self):
        with self.output().open("w") as f:
            f.write(str(0))

    def requires(self):
        return Extract_from_nc_images()

    def output(self):
        return luigi.LocalTarget("tmp/imgs.txt")

