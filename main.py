import numpy as np
import cv2
import os
from loader import Loader
from matcher import Matcher

match = Matcher()
load = Loader()

if __name__=="__main__":
    folder = "./data/2"
    images = load.load_images(folder)
    pts1, pts2 = match.find_matches(images[0],images[1])





