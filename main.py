import numpy as np
import cv2
import os
from loader import Loader
from matcher import Matcher
from numpy import linalg
from scipy.linalg import svd
from compute import Compute

match = Matcher()
load = Loader()
comp = Compute()



if __name__=="__main__":
    folder = "./data/1"
    images = load.load_images(folder)
    pts1, pts2 = match.find_matches(images[0],images[1])
    H = comp.compute_H(pts1, pts2)
    print(H)
    i = comp.stitch_images(images[0], images[1], H)
    cv2.imshow("stich",i)
    cv2.waitKey(0)




