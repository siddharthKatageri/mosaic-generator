import numpy as np
import cv2
import os
from loader import Loader


load = Loader()

if __name__=="__main__":
    folder = "./data/1"
    images = load.load_images(folder)
    cv2.imshow("i", images[0])
    cv2.waitKey(0)


    #assert img1.shape==img2.shape, "image shape do not match"


