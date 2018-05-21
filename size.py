import cv2
import numpy as np

image = cv2.imread("img/s3.jpg")

# kernel = np.ones((5,5), np.uint8)
# tmp = cv2.erode(image, kernel, iterations=1)
# tmp = cv2.dilate(image, kernel, iterations=1)

# blur = cv2.medianBlur(image, 11)
gaussianBlur = cv2.GaussianBlur(image, (7,7), 0)
gray = cv2.cvtColor(gaussianBlur, cv2.COLOR_BGR2GRAY)

edges = cv2.Canny(gray, 50, 150, apertureSize=3)

# image, contours, hierarchy = cv2.findContours(edges,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
# cv2.drawContours(image, contours, -1, (255, 0, 0), 3)

# tmp_edges = cv2.medianBlur(edges, 2)
lines = cv2.HoughLines(edges, 1, np.pi/180, 80)
allLines = []
tmp_lines = lines[:, 0, :]
for rho, theta in tmp_lines[:]:
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 2000*(-b))
    y2 = int(y0 - 2000*(a))
    cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 1)
    allLines.append([[x1, y1], [x2, y2]])
print(len(allLines))

# lines = cv2.HoughLinesP(edges, 1, np.pi/180, 30, minLineLength=30, maxLineGap=20)
# line_count = 0
# tmp_lines = lines[:, 0, :]
# for x1, y1, x2, y2 in tmp_lines:
#     cv2.line(image, (x1, y1), (x2, y2), (0, 255, 2), 1)
#     line_count = line_count + 1
# print(line_count)

imgray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(imgray,127,255,0)
im, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(im, contours, -1, (255, 0, 0), 3)


def writeText(img, text, org):
    font                   = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
    bottomLeftCornerOfText = org
    fontScale              = 0.5
    fontColor              = (0,0,0)
    lineType               = 1

    cv2.putText(img, text,
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                lineType)

writeText(image, '37.5cm', (10,100))

cv2.imshow('image', image)
# cv2.imshow('gray', gray)
cv2.imshow('tmp', im)
cv2.imshow('edges', edges)

# laplacian = cv2.Laplacian(gray, cv2.CV_16S, ksize=3)
# dst = cv2.convertScaleAbs(laplacian)
# cv2.imshow('laplacian', dst)

cv2.waitKey(0)
cv2.destroyAllWindows()
