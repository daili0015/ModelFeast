
import pydicom
import scipy.misc
import numpy as np
from PIL import Image
import os, math
import cv2 


def bbox(mask):
    _, binary = cv2.threshold(mask, 0.5, 255, cv2.THRESH_BINARY)  
    _, contours, _ = cv2.findContours(binary,cv2.RETR_EXTERNAL, \
        cv2.CHAIN_APPROX_SIMPLE)
    assert len(contours)==1
    x,y,w,h = cv2.boundingRect(contours[0])
    return x,y,w,h

def is_invert_folder(folder):
    files = [f for f in os.listdir(folder) if ".dcm" in f]
    files.sort(key=str)
    ds1, ds2 = pydicom.read_file(os.path.join(folder, files[0])), \
        pydicom.read_file(os.path.join(folder, files[1]))
    invert_order = 1 if ds1.ImagePositionPatient[2] < ds2.ImagePositionPatient[2] else 0
    return invert_order

def normalize_hu(image):
    MIN_BOUND = -115
    MAX_BOUND = 235
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image > 1] = 1.
    image[image < 0] = 0.

    # print( np.mean(image), np.std(image) )
    # print( image.shape, image.size )
    return image.astype(np.float32)

def mk_dir(path):
    if not os.path.exists(os.path.join(path)):
        os.makedirs(os.path.join(path))
        
def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices])
    image = image.astype(np.int16)
    # print(np.max(image), np.min(image))
    image[image == -2000] = 0
    for slice_number in range(len(slices)):
        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope
        if slope != 1:
            print("slope != 1")
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)
        image[slice_number] += np.int16(intercept)
    return np.array(image, dtype=np.int16)
