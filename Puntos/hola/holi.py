import cv2
import imutils

image = cv2.imread('ardilla.jpg')

# Escalando una imagen usando imutils.resize
imageOut = imutils.resize(image,width=300)

cv2.imshow('Imagen de entrada',image)
cv2.imshow('Imagen de salida',imageOut)
cv2.waitKey(0)
cv2.destroyAllWindows()