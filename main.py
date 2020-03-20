import numpy as np
import cv2
import os
from loader import Loader
from matcher import Matcher
from numpy import linalg
from scipy.linalg import svd
from compute import Compute


match = Matcher()
load = Loader(0.3)
comp = Compute()



if __name__=="__main__":
    folder = "./data/5"
    saveto = str(folder[-1])+"/"
    images = load.load_images(folder)
    print("Image shape:",images[0].shape)

    I2 = images[1]
    print("total images in folder:",len(images))
    for i in range(1, len(images)+1):
        I1 = images[i - 1]

        # Finding matches between the two images
        pts1, pts2 = match.find_matches(I1, I2)
        H = comp.compute_H(pts1, pts2)
        print("COmputed Homography:"+"("+str(i-1)+"-"+str(i)+")\n",H)
        I2 = comp.stitch_images(I1, I2, H)
        cv2.imwrite("./results/"+saveto+str(i)+".jpg",I2)
    cv2.imshow("stiched", I2)
    cv2.waitKey(0)



