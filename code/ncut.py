import cv2

from skimage import data, segmentation, color
from skimage.future import graph
from matplotlib import pyplot as plt

# img = data.coffee()

img = cv2.imread('ncut_test.jpg', cv2.IMREAD_COLOR)
img = cv2.resize(img, (int(1128 / 2), int(720 / 2)), 0, 0, cv2.INTER_LINEAR)
labels1 = segmentation.slic(img, compactness=30, n_segments=800)
print("--- 11")
out1 = color.label2rgb(labels1, img, kind='avg')
print("--- 22")

fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True, figsize=(6, 8))

ax[0].imshow(img)
ax[1].imshow(out1)

for a in ax:
    a.axis('off')

plt.tight_layout()
plt.show()
