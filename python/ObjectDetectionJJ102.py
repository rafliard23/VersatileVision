import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# Load image jj-102

img_scene = cv.imread('jj102.jpg')
assert img_scene is not None, "File could not be read, check with os.path.exists()"

img_show = cv.cvtColor(img_scene, cv.COLOR_BGR2RGB)
cv.imshow('Testing', img_show)

# Process pencarian contour

img_gray = cv.cvtColor(img_scene, cv.COLOR_BGR2GRAY)
ret, threshold = cv.threshold(img_gray, 32, 255, 0)

cv.imshow('Thresholding', threshold)

cv.waitKey(0)

cv.destroyAllWindows()