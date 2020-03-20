import numpy as np
import cv2
from scipy.linalg import svd

class Compute:
    def __init__(self):
        None

    #solve the Ax=0 using svd
    def compute_H(self,imagePoints,imagePoints_dash):
        self.imagePoints = imagePoints
        self.imagePoints_dash = imagePoints_dash
        total=len(self.imagePoints_dash)

        A=np.zeros((2*total,9))        #initialize a zero matrix of (2n*9)

        row1=[]
        row2=[]
        xvalue=0

        #for w in zip(imagePoints,imagePoints_dash):      #this for loop computes and builds the matrix A
        for w in zip(self.imagePoints_dash,self.imagePoints):
            row1.append(w[0][0])
            row1.append(w[0][1])
            row1.append(1)
            row1.append(0)
            row1.append(0)
            row1.append(0)
            row1.append(int(-w[1][0]*w[0][0]))
            row1.append(int(-w[1][0]*w[0][1]))
            row1.append((-w[1][0]))


            row2.append(0)
            row2.append(0)
            row2.append(0)
            row2.append(w[0][0])
            row2.append(w[0][1])
            row2.append(1)
            row2.append(int(-w[1][1]*w[0][0]))
            row2.append(int(-w[1][1]*w[0][1]))
            row2.append((-w[1][1]))

            for j in range(9):
                A[xvalue][j]=row1[j]

            xvalue+=1
            for j in range(9):
                A[xvalue][j]=row2[j]
            xvalue+=1
            row1=[]
            row2=[]
    #------------------
        u, s, v = svd(A)
        H = v[-1]
        H = np.reshape(H,(3,3))
        H = H/H[2,2]
        return(H)

    def stitch_images(self,I1, I2, H):
        """
        Stitch two images I1 and I2 using homography.
        Input:
            I1 = Image 1
            I2 = Image 2
            H = Homography matrix
        Output:
            I_1 = Stitched image
        """
        self.I1 = I1
        self.I2 = I2
        self.H = H
        r = I1.shape[0]
        c = I1.shape[1]

        I = np.zeros((self.I2.shape[0] + 2 * self.I1.shape[0], self.I2.shape[1] + 2 * self.I1.shape[1], 3), dtype=np.uint8)
        I[self.I1.shape[0]:self.I1.shape[0] + self.I2.shape[0], self.I1.shape[1]:self.I1.shape[1]+self.I2.shape[1], :] = self.I2
        I = I.astype(int)

        # i = 0
        for row_idx in range(I.shape[0]):
            for col_idx in range(I.shape[1]):
                x = self.H @ np.array([[col_idx - c, row_idx - r, 1]]).T
                x = np.floor(x / x[2]).astype(int)
                if (0 <= x[1] and x[1] <= (r - 1) and 0 <= x[0] and x[0] <= c - 1):
                    I[row_idx, col_idx, :] = I1[x[1], x[0], :]

        # Crop out blackened region
        I_gray = np.sum(I.astype(int), axis=2) / 3
        axis = np.array(np.where(I_gray != 0)).T
        minR = np.min(axis[:, 0])
        maxR = np.max(axis[:, 0])
        minC = np.min(axis[:, 1])
        maxC = np.max(axis[:, 1])
        I_1 = np.uint8(I[minR:maxR, minC:maxC])

        return I_1
