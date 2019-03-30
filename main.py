import time
import logging
from logging.config import dictConfig
from skimage import io, morphology, exposure, color
from inPainting import *
import cv2

def main():
    filePhoto = "img/photo.jpg"
    fileMask = "img/mask.jpg"
    img_grayscale = color.rgb2gray(io.imread(filePhoto))
    img_mask = color.rgb2gray(cv2.imread(fileMask))
    img = inPainting(img_grayscale,img_mask)


if __name__ == '__main__':
    main()
