import cv2
import numpy as np

# img = cv2.imread('test.png',cv2.IMREAD_GRAYSCALE)
img = cv2.imread('IMG_0825.jpeg',cv2.IMREAD_GRAYSCALE)
cv2.imshow('image',img)
cv2.waitKey(1000)
cv2.destroyAllWindows()