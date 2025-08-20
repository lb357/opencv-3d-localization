import cv2
import numpy as np
import os
import glob

CHECKERBOARD = (6, 9)
subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
calibration_flags = 0
objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
_img_shape = None
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

# Create calib directory if it doesn't exist
if not os.path.exists('calib'):
    os.makedirs('calib')

images = glob.glob('images/*.jpg')
print(len(images))
for fname in images:
    img = cv2.imread(fname)
    if _img_shape is None:
        _img_shape = img.shape[:2]
    else:
        assert _img_shape == img.shape[:2], "All images must share the same size."
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
    print(fname, ret)
    if ret:
        objpoints.append(objp)
        cv2.cornerSubPix(gray, corners, (3, 3), (-1, -1), subpix_criteria)
        imgpoints.append(corners)

        # Draw and display the corners
        cv2.drawChessboardCorners(img, CHECKERBOARD, corners, ret)

        # Save the image with keypoints to 'calib' directory
        output_path = os.path.join('calib', os.path.basename(fname))
        cv2.imwrite(output_path, img)

N_OK = len(objpoints)

rms, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],
                                                   None,None,
                                                   flags = calibration_flags,
                                                   criteria = (
                                                       cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6
                                                   )
    )
print("Found " + str(N_OK) + " valid images for calibration")
print("DIM=" + str(_img_shape[::-1]))
print("cameraMatrix=np.array(" + str(mtx.tolist()) + ")")
print("distCoeffs=np.array(" + str(dist.tolist()) + ")")

with open('param.txt', 'w') as f:
    f.write("np.array(" + str(mtx.tolist()) + ")\n")
    f.write("np.array(" + str(dist.tolist()) + ")")
