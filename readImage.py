import cv2
import os
import glob
import math

img_dir = "/Users/macbookpro/desktop/1178"
data_path = os.path.join(img_dir, '*g')
files = glob.glob(data_path)

data = []
for f1 in files:
    img = cv2.imread(f1)
    data.append (img)

