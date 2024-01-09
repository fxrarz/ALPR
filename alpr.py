import pytesseract
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

#detecting license plate on the vehicle
plateCascade = cv2.CascadeClassifier('indian_license_plate.xml')

file_path = "/content/images/cover_3.jpeg"

def plate_detect(img):
	plateImg = img.copy()
	roi = img.copy()
	plateRect = plateCascade.detectMultiScale(plateImg,scaleFactor = 1.2, minNeighbors = 7)
	
	for (x,y,w,h) in plateRect:
		roi_ = roi[y:y+h, x:x+w, :]
		plate_part = roi[y:y+h, x:x+w, :]
		cv2.rectangle(plateImg,(x+2,y),(x+w-3, y+h-5),(0,255,0),3)
	return plateImg, plate_part, ((x+2,y),(x+w-3, y+h-5))

def display_img(img):
	img_ = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
	plt.imshow(img_)
	plt.show()

#test image is used for detecting plate
inputImg = cv2.imread(file_path)
inpImg, plate, loc = plate_detect(inputImg)
display_img(inpImg)
display_img(plate)

print('masha allha')