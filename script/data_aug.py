import xml.etree.ElementTree as ET
import os.path
import os
from glob import glob

import numpy as np
import cv2
import sys
import random
from shutil import copyfile


def augment_brightness(image):
    #cv2.imwrite("ori.png", image)
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    random_bright = .15+np.random.uniform()
    #print(random_bright)
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    #cv2.imwrite("after.png", image1)
    return image1

def darker_img(image):
    # Convert to YUV
    img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    img_gray = img_yuv[:,:,0]
   
    # Pick the majority pixels of the image
    idx = (img_gray<245) & (img_gray > 10)
    
    # Make the image darker
    img_gray_scale = img_gray[idx]*np.random.uniform(0.1,0.4)
    img_gray[idx] = img_gray_scale
    
    # Convert back to BGR 
    img_yuv[:,:,0] = img_gray
    img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    return img

def add_random_shadow(image):
    top_y = image.shape[1]*np.random.uniform()
    top_x = 0
    bot_x = image.shape[0]
    bot_y = image.shape[1]*np.random.uniform()
    image_hls = cv2.cvtColor(image,cv2.COLOR_RGB2HLS)
    shadow_mask = 0*image_hls[:,:,1]
    X_m = np.mgrid[0:image.shape[0],0:image.shape[1]][0]
    Y_m = np.mgrid[0:image.shape[0],0:image.shape[1]][1]
    
    shadow_mask[((X_m-top_x)*(bot_y-top_y) -(bot_x - top_x)*(Y_m-top_y) >=0)]=1
    random_bright = .15+.8*np.random.uniform()
    if np.random.randint(2)==1:
    #    random_bright = .5
        cond1 = shadow_mask==1
        cond0 = shadow_mask==0
        if np.random.randint(2)==1:
            image_hls[:,:,1][cond1] = image_hls[:,:,1][cond1]*random_bright
        else:
            image_hls[:,:,1][cond0] = image_hls[:,:,1][cond0]*random_bright    
    image = cv2.cvtColor(image_hls,cv2.COLOR_HLS2RGB)
    return image


# Go through dataset dir
# Rename file path of xml files

directory = "../data/dataset_xml/"
xml_list = glob(directory+'*.xml') 

for f in xml_list:

    flip_filename = f[0:-4]+'_flip.xml'
    print 'copy ', f, flip_filename

    # Copy as flip.xml
    copyfile(f, flip_filename)

    # ---------------------
    # XML processing
    # ---------------------

    # Read the xml file just copied
    tree = ET.parse(flip_filename)
    root = tree.getroot()
    state = root.find('object').find('name').text

    # Only flip image is it is not 'off' class/category image
    if state == 'off':
        print 'remove', flip_filename
        os.remove(flip_filename)
    else:
        filename = f[-14:-4]+'_flip' # Add '_flip' to filename
        print 'do the job, ', filename
    
        # Fill in folder/filename/path etc contents
        root.find('folder').text = "dataset_jpg"
        root.find('filename').text = filename+'.jpg'
        root.find('path').text = "dataset_jpg_flip/"+filename+'.jpg'
    
        # label bounding box coordinates
        x1 = int(root.find('object').find('bndbox').find('xmin').text)
        y1 = int(root.find('object').find('bndbox').find('ymin').text)
        x2 = int(root.find('object').find('bndbox').find('xmax').text)
        y2 = int(root.find('object').find('bndbox').find('ymax').text)
        width = int(root.find('size').find('width').text)
        height = int(root.find('size').find('height').text)
        y1n = y1
        x1n = width -x2
        x2n = width - x1
        y2n = y2
        root.find('object').find('bndbox').find('xmin').text = str(x1n)
        root.find('object').find('bndbox').find('xmax').text = str(x2n)
    
        # Write updated contents out
        tree.write(flip_filename)
    
        # ---------------------
        # JPG processing
        # ---------------------
        img = cv2.imread('../data/dataset_jpg/'+filename[-1:-5]+'.jpg')
        img_flip = cv2.flip(img, 1) # 1 - horizontal flip
        cv2.imwrite('../data/dataset_jpg_flip/'+filename+'.jpg', img_flip)


