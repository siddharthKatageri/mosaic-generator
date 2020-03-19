import cv2
import numpy as np

class Matcher:
    def __init__(self):
        self.sift = cv2.xfeatures2d.SIFT_create()
        self.bf = cv2.BFMatcher()

    #finding good matches
    def find_matches(self,img1, img2):
        self.img1 = img1
        self.img2 = img2
        i1 = cv2.cvtColor(self.img1, cv2.COLOR_RGB2GRAY)
        i2 = cv2.cvtColor(self.img2, cv2.COLOR_RGB2GRAY)


        kp1,d1 = self.sift.detectAndCompute(i1,None)
        kp2,d2 = self.sift.detectAndCompute(i2,None)

        matches = self.bf.knnMatch(d1,d2,k=2)

        good=[]
        for m,n in matches:
            if m.distance < 0.5*n.distance:
                good.append([m])

        m1=[]
        m2=[]
        for mat in good:
            img1_idx = mat[0].queryIdx
            img2_idx = mat[0].trainIdx


            m1.append(kp1[img1_idx].pt)
            m2.append(kp2[img2_idx].pt)

        m1 = np.array(m1)
        m2 = np.array(m2)
        return m1, m2
