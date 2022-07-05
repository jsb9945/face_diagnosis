#!/usr/bin/env python

from tkinter import W
import cv2
import numpy as np

cap = cv2.VideoCapture(2)                       # 2번 카메라 연결

if cap.isOpened():
    while True:
        ret, frame = cap.read()                 # 카메라 프레임 읽기
        if ret:
            cv2.imshow('camera', frame)          # 프레임 화면에 표시
            if cv2.waitKey(1) != -1:            # 아무 키나 누르면 종료
                cv2.imwrite('photo.jpg', frame)  # 프레임을 'photo.jpg'에 저장
                break
        else:
            print('no frame!')
            break
else:
    print('no camera!')
cap.release()

image = cv2.imread('photo.jpg', cv2.IMREAD_UNCHANGED)  # 사진 원본 사용
# img = cv2.imshow('image', image)

img_gau = cv2.GaussianBlur(image, (0,0), 6)  # 가우시안 필터 적용
# cv2.imshow("img_gau", img_gau)

img_bitwise_not_bgr = cv2.bitwise_not(img_gau)  # 보색
# cv2.imshow("img_bitwise_not_bgr", img_bitwise_not_bgr)

img_bitwise_not_bgr2gray = cv2.cvtColor(img_bitwise_not_bgr, cv2.COLOR_BGR2GRAY)  # 보색으로 된 사진을 그레이 스케일로 변경
# cv2.imshow("img_bitwise_not_bgr2gray", img_bitwise_not_bgr2gray)

ret, img_binary = cv2.threshold(img_bitwise_not_bgr2gray, 150,255,cv2.THRESH_BINARY)
# cv2.imshow("img_binary", img_binary)

contours, hierarchy = cv2.findContours(img_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
img_contour = cv2.drawContours(image, contours, -1, (0, 255, 0), 2)
cv2.imshow("img_contour", img_contour)

BLUR = 21
CANNY_THRESH_1 = 18
CANNY_THRESH_2 = 28
MASK_DILATE_ITER = 2
MASK_ERODE_ITER = 2
MASK_COLOR = (0, 0, 0)  # In BGR format

img = img_contour
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

edges = cv2.Canny(gray, CANNY_THRESH_1, CANNY_THRESH_2)
edges = cv2.dilate(edges, None)
edges = cv2.erode(edges, None)

contour_info = []
contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

for c in contours:
    contour_info.append((
        c,
        cv2.isContourConvex(c),
        cv2.contourArea(c),
    ))

contour_info = sorted(contour_info, key=lambda c: c[2], reverse=True)
max_contour = contour_info[0]

mask = np.zeros(edges.shape)
cv2.fillConvexPoly(mask, max_contour[0], (255))

mask = cv2.dilate(mask, None, iterations=MASK_DILATE_ITER)
mask = cv2.erode(mask, None, iterations=MASK_ERODE_ITER)
mask = cv2.GaussianBlur(mask, (BLUR, BLUR), 0)
mask_stack = np.dstack([mask] * 3)

mask_stack = mask_stack.astype('float32') / 255.0
img = img.astype('float32') / 255.0

masked = (mask_stack * img) + ((1 - mask_stack) * MASK_COLOR)
masked = (masked * 255).astype('uint8')

dst = cv2.resize(masked, dsize=(640, 480), interpolation=cv2.INTER_AREA)

cv2.imshow("dst", dst)

cv2.waitKey(0)

cv2.destroyAllWindows()
