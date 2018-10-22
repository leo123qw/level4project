# -*- coding: UTF-8 -*-
import skimage.io as io
from skimage import data_dir
import numpy as np
import argparse
import cv2
import os


# #detect red
# color = [
# 	([0,0,153],[102,102,255])# order in b,g,r is red range.
# ]
# for (lower,upper) in color:
# 	#create a numpy array
# 	lower = np.array (lower, dtype= "uint8") #lower limit for the red color
# 	upper = np.array (upper, dtype= "uint8") #upper limit for the red color

# 	#find the color denpend on the threahold.
# 	nki_coll_he_image_collection=[]
# 	for j in range(len(nki_coll_he)):
# 		nki_coll_he_image = nki_coll_he[j]
# 		mask = cv2.inRange(nki_coll_he_image,lower,upper)
# 		output = cv2.bitwise_and(nki_coll_he_image, nki_coll_he_image, mask = mask)
# 		nki_coll_he_image_collection.append(nki_coll_he_image)
# cv2.imshow("images",np.hstack([nki_coll_he_image, output]))
# cv2.waitKey(0)


#readImage into python
#nki_traiig and vgh_traing need to be adjust when somepeople download from github.
nki_training = '/Users/macbookpro/desktop/level4project/images/EpiStromaTrainingImages/NKI_Training/*.jpg'
vgh_training = '/Users/macbookpro/desktop/level4project/images/EpiStromaTrainingImages/VGH_Training/*.jpg'
nki_coll = io.ImageCollection(nki_training)
vgh_coll = io.ImageCollection(vgh_training)
print (len(nki_coll)) #214 pics
print (len(vgh_coll)) #102 pics


#separate the collection into he_image and normal image
def nki_coll_he():
	nki_coll_he=[]
	for i in range (len(nki_coll)/2,len(nki_coll)):
		nki_he_image = nki_coll[i]
		#res=cv2.resize(nki_he_image,None,fx=0.3,fy=0.3,interpolation=cv2.INTER_CUBIC)
		nki_coll_he.append(nki_he_image)
		# cv2.imshow('m',res)
	return nki_coll_he
	
def vgh_coll_he():
	vgh_coll_he=[]
	for i in range (len(vgh_coll)/2,len(vgh_coll)):
		vgh_he_image = vgh_coll[i]
		#res=cv2.resize(vgh_he_image,None,fx=0.3,fy=0.3,interpolation=cv2.INTER_CUBIC)
		vgh_coll_he.append(vgh_he_image)
		
	return vgh_coll_he

def resize():
	he_coll_resize=[]
	he_coll = nki_coll_he + vgh_coll_he
	for i in he_coll:
		res=cv2.resize(i,None,fx=0.3,fy=0.3,interpolation=cv2.INTER_CUBIC)
		he_coll_resize.append(res)

	return he_coll_resize

#----------------------main --------------------------------------
nki_coll_he = nki_coll_he()
vgh_coll_he = vgh_coll_he()

he_coll_resize = resize()

red=np.uint8([[[255,0,0]]]) #bgr
hsv_red=cv2.cvtColor(red,cv2.COLOR_BGR2HSV)
print (hsv_red)
for img in he_coll_resize:
	hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
	lowper_red= np.array([102, 35,0])
	upper_red=np.array([255,248,248])
	mask=cv2.inRange(hsv,lowper_red,upper_red)
	res=cv2.bitwise_and(img,img,mask=mask)
cv2.imshow('Image',img)
cv2.imshow('mask',mask)
cv2.waitKey(0)
