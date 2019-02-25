#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: zcy
# @Date:   2019-02-16 11:06:01
# @Last Modified by:   zcy
# @Last Modified time: 2019-02-24 09:55:58


import numpy as np 
import os, cv2

def edge_demo(image):
    blurred = cv2.GaussianBlur(image, (3, 3), 0)
    gray = cv2.cvtColor(blurred, cv2.COLOR_RGB2GRAY)
    t = 30
    edge_output = cv2.Canny(gray, t, t*3, apertureSize=3, L2gradient=True)

    cv2.imwrite('edge_output.png', edge_output)

src = cv2.imread('0033.png')
edge_demo(src)

