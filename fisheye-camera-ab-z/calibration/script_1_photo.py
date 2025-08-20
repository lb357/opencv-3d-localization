import cv2
import os
import time

CAM_ID = 1
CAM_RESOLUTION = (3840, 2160)


cv2.namedWindow("preview")
cam = cv2.VideoCapture(CAM_ID, cv2.CAP_DSHOW)
cam.set(cv2.CAP_PROP_FRAME_WIDTH,CAM_RESOLUTION[0])
cam.set(cv2.CAP_PROP_FRAME_HEIGHT,CAM_RESOLUTION[1])


if cam.isOpened():
    read_ok, image = cam.read()
    if not os.path.isdir('/images'):#проверяем наличие папки
        os.mkdir ('/images')# если нет, создаем
else:
    read_ok = False

while read_ok:
    cv2.imshow("preview", cv2.resize(image, (image.shape[1]//2, image.shape[0]//2)))
    read_ok, image = cam.read()
    key = cv2.waitKey(20)
    if key == 32:
        cv2.imwrite(f"images/{time.time()}.jpg", image)
        print(f"File: images/{time.time()}.jpg saved")
    if key == 27: # exit on ESC
        break

cam.release()
cv2.destroyWindow("preview")
