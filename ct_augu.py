#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: zcy
# @Date:   2019-02-11 11:53:24
# @Last Modified by:   zcy
# @Last Modified time: 2019-02-20 10:45:47

import numpy as np 
import cv2, math

def show(images):
    print(np.mean(images), np.std(images))
    print(images.dtype)
    cv2.imshow('image', images[3])
    cv2.waitKey(0)

def random_flip(img, p=0.5):
    isflip = np.random.randint(0, 100)/100.0 < p
    if isflip:
        # img = np.fliplr(img.copy())
        img = np.flip(img.copy(), 1)
    return img

def random_flip2(imgs, p=0.5):
    isflip = np.random.randint(0, 100)/100.0 < p
    if isflip:
        for i in range(imgs.shape[0]):
            imgs[i] = np.flip(imgs[i].copy(), 1)
    return imgs    

def random_rotate(img, max_angle):
    height = img.shape[0]
    width = img.shape[1]
    scale = 1
    angle = np.random.randint(-max_angle, max_angle)
    rotateMat = cv2.getRotationMatrix2D((width/2,height/2),angle,scale)
    rotateImg = cv2.warpAffine(img,rotateMat,(width,height), \
        flags=cv2.INTER_LINEAR, borderValue=0.0)
    # print(np.mean(img), np.std(img))
    # print(np.mean(rotateImg), np.std(rotateImg))
    # print(img.dtype, rotateImg.dtype)
    return rotateImg

def resize_np(images_zyx, desire_dim, desire_size, angle=0,\
            flip=0, verbose=False):
    if verbose:
        print("Shape: ", images_zyx.shape)
    # print "Resizing dim z"
    res = cv2.resize(images_zyx, dsize=(images_zyx.shape[1], desire_dim), interpolation=cv2.INTER_LINEAR)
    if verbose:
        print("after resize dim: Shape is ", res.shape)

    # resize w h 
    res = res.swapaxes(0, 2)
    res = res.swapaxes(0, 1)
    assert res.shape[2] < 513
    # yxz
    res = cv2.resize(res, dsize=desire_size, interpolation=cv2.INTER_LINEAR)
    # rotate
    if angle:
        # res = random_rotate(res, angle)
        pass

    res = res.swapaxes(0, 2)
    res = res.swapaxes(2, 1)
    
    # flip
    if flip:
        res = random_flip2(res, p=flip)

    if verbose:
        print("after resize w and h: Shape is ", res.shape)

    return res

def RandomCropResize(image_zyx, ratio_range=0.8):
    '''crop [ratio, 1] from origin data'''
    ratio = np.random.randint(ratio_range*100, 100)/100.0
    ori_shape = image_zyx.shape
    z_length = int(ori_shape[0]*ratio)
    y_length = int(ori_shape[1]*ratio)
    x_length = int(ori_shape[2]*ratio)
    # get zyx start position
    z_start = np.random.randint(0, ori_shape[0]-z_length)
    y_start = np.random.randint(0, ori_shape[1]-y_length)
    x_start = np.random.randint(0, ori_shape[2]-x_length)
    # print(ratio, z_start, y_start, x_start)
    image = image_zyx[z_start: z_start + z_length,
                      y_start: y_start + y_length,
                      x_start: x_start + x_length]
    return image

def RandomCrop(image_zyx, crop_pixels=5):

    ori_shape = image_zyx.shape
    y_length = int(ori_shape[1])-crop_pixels
    x_length = int(ori_shape[2])-crop_pixels

    y_start = np.random.randint(0, crop_pixels)
    x_start = np.random.randint(0, crop_pixels)
    image = image_zyx[:,
                      y_start: y_start + y_length,
                      x_start: x_start + x_length]

    return image 

def Resize(image_zyx, size=(80, 128, 128), flip=0, angle=0):
    '''crop [ratio, 1] from origin data'''
    image = resize_np(image_zyx, size[0], size[1:], \
         angle = angle, flip=flip, verbose = False)
    return image


if __name__ == '__main__':
    a = np.load('/SSD/data/train_norm/0A2E9075-5C56-4E0D-AA2F-383002345D7C/norm_data.npy')
    print(a.shape)
    b = RandomCrop(a, crop_pixels=5)  
    print(b.shape)
    # while True:
    #     b = Resize(a, flip=0.5) 
    # b = Resize(a)  
    # print(b.shape)

