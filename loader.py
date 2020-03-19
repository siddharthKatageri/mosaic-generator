import cv2
import os


class Loader:
    def __init__(self):
        None

    #scaling down images by given factor
    def scale_down(self, img, factor):
        self.img = img
        self.factor = factor
        return cv2.resize(self.img, (int(self.img.shape[1]*self.factor),int(self.img.shape[0]*self.factor)))

    #load images
    def load_images(self, folder):
        self.folder = folder
        images = []
        for iname in os.listdir(self.folder):
            img = cv2.imread(os.path.join(self.folder,iname))
            if img is not None:
                img = self.scale_down(img, 0.5)
                images.append(img)
        return images
