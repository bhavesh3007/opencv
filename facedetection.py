import cv2 as cv
import numpy as np


# read the image 
img = cv.imread('C:/Users/chouh/OneDrive/Desktop/opencv/image/1.jpg')
hight, width = img.shape[0], img.shape[1]

# resize VM if required 
img=cv.resize(img, (int(width * 0.4), int(hight * 0.4)), interpolation=cv.INTER_AREA )
cv.imshow('bhavesh', img)

# convert image to gray
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#cv.imshow('gray ', gray)

# haarcascade classifier read it to var
haarcasface= cv.CascadeClassifier('C:/Users/chouh/OneDrive/Desktop/opencv/haarcascade.xml')
haarcaseye= cv.CascadeClassifier('C:/Users/chouh/OneDrive/Desktop/opencv/haarcascadeeye.xml')

# get the pairs of cordinates of face and eyes
faces_rect= haarcasface.detectMultiScale(gray, 1.1, 2)
eye_rect= haarcaseye.detectMultiScale(gray, 1.1, 1)

# count to pairs of condinates to count faces and eyes
print(f'count of faces: {len(faces_rect)}')
print(f'count of eyes: {len(eye_rect)}')


# draw rectangle over face and eyes using pair of condinates
for (x,y,g,h) in faces_rect:
    cv.rectangle(img, (x,y), (x+g, y+h), (0,255,0), 1)

for (x,y,g,h) in eye_rect:
    cv.rectangle(img, (x,y), (x+g, y+h), (0,0,255), 1)

#show image
cv.imshow('faces detected', img)
cv.waitKey(0)