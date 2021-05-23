# -*- coding: utf-8 -*-
"""
Created on Sat May  8 11:01:36 2021

@author: Safwen
"""

import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def rotate_img(img, rot):
    if rot==0:
        return(img)
    elif rot==90:
        return img.rotate(90)
    elif rot==120:
        return img.rotate(120)
    elif rot==180:
        return img.rotate(180)
    elif rot==240:
        return img.rotate(240)
    elif rot==270:
        return img.rotate(270)
    else:
        raise ValueError('ERROR: False rotation')
    