import skimage.io as io
from skimage import data_dir
import numpy as np
import argparse
import cv2
import os
#readImage into python
# nki_traiig and vgh_traing need to be adjust when somepeople download from github.
# nki_training = '/Users/macbookpro/desktop/level4project/EpiStromaTrainingImages/NKI_Training/*.jpg'
# vgh_training = '/Users/macbookpro/desktop/level4project/EpiStromaTrainingImages/VGH_Training/*.jpg'
# nki_coll = io.ImageCollection(nki_training)
# vgh_coll = io.ImageCollection(vgh_training)
# print (len(nki_coll)) #214 pics
# print (len(vgh_coll)) #102 pics
# io.imshow(nki_coll[107])
# io.show()

#separate the collection into he_image and normal image
# nki_coll_he=[]
# for i in range (len(nki_coll)/2,len(nki_coll)):
# 	nki_he_image = nki_coll[i]
# 	nki_coll_he.append(nki_he_image)
# print (nki_coll_he)

# vgh_coll_he=[]
# for i in range (len(vgh_coll_he)/2,len(vgh_coll_he)):
# 	vgh_he_image = vgh_coll[i]
# 	vgh_coll_he.append(vgh_he_image)
# print (vgh_coll_he)

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

# def detect_color(img_path, mark_img_path):
# 	"""" detect the different color region from the same image  """
# 	image = cv2.imread(img_path) #load images
# 	boundaries = [
# 	([0,0,255],[0,0,255]), # red
# 	([0,255,0],[0,255,0]), # green
# 	([255,0,0],[255,0,0]), # blue
# 	]
# 	#look through the color range
# 	for (lower, upper) in boundaries:
# 		#create a numpy array based on the color range
# 		lower = np.array(lower, dtype = "uint8")
# 		upper = np.array(upper, dtype = "uint8")

# 		# create a mask based on specific color range
# 		mask =cv2.inRange(image, lower, upper)
# 		output = cv2.bitwise_and(image, image, mask = mask)


# 		mark_zone_with_color(output, mark_img_path, lower)
# def mark_zone_with_color(src_img, mark_img, mark_color):

# 	# convert to gray color
# 	gray = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)

# 	ret, binary =  cv2,threahold(gray, 0, 255, cv2.THRESH_BINARY) #just scrach the contour

# 	#contour detection
# 	_,contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

# 	newImg = cv2.imread(mark_img)
# 	newImg = cv2.resize(newImg, (512, 512))

# 	#drawing
# 	for i in range (len(contours) - 1):
# 		cv2.drawContours(image = newImg, contours = contours[i+1], contourIdx= -1, color = tuple(mark_color.tolist()), thickness = 2, maxlevel = 1, linetype = 8)

# 	cv2.imwrite(mark_img, newImg)

# def batch_marker(src_img_dir, draw_contours_img_dir):
# 	src_imgs = get_filenames_in_dir(src_img_dir)
# 	dc_imgs = get_filenames_in_dir(draw_contours_img_dir)

# 	for src in src_imgs:
# 		for dc in dc_imgs:
# 			if src == dc:
# 				detect_color(os.path.join(src_img_dir, src), os.path.join(draw_contours_img_dir, dc))
 
# def get_filenames_in_dir(dir):
# 	""" get all of the file name in the same dir """
# 	for root, dirs, files in os.walk(dir):
# 		return files


red=np.uint8([[[0,0,255]]])
hsv_red=cv2.cvtColor(red,cv2.COLOR_BGR2HSV)
print hsv_red
img=cv2.imread("/Users/macbookpro/desktop/level4project/EpiStromaTrainingImages/NKI_Training/3_114_2_5.jpg")
res=cv2.resize(img,None,fx=0.3,fy=0.3,interpolation=cv2.INTER_CUBIC)
cv2.imshow('m',res)
hsv=cv2.cvtColor(res,cv2.COLOR_BGR2HSV)
lower_red=np.array([170,100,100])
upper_red=np.array([180,255,255])
mask=cv2.inRange(hsv,lower_red,upper_red)
res=cv2.bitwise_and(res,res,mask=mask)
cv2.imshow('frame',res)
cv2.imshow('mask',mask)
cv2.waitKey(0)
