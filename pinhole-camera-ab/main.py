import cv2
import numpy as np
import camutil
import re3d

CAM_ID = 1
TARGET_ARUCO_IDS = [80]
TARGET_ARUCO_SIZE = 0.06

CAM_RESOLUTION = (1920, 1080)
CAM_MAT_A = 1
CAM_AXES_CUBE_SIZE = 0.350
CAM_AXES_POINTS = [
    np.array([0.000, 0.000, 0.000, 1], dtype=np.float64),
    np.array([CAM_AXES_CUBE_SIZE, 0.000, 0.000, 1], dtype=np.float64),
    np.array([CAM_AXES_CUBE_SIZE, CAM_AXES_CUBE_SIZE, 0.000, 1], dtype=np.float64),
    np.array([0.000, CAM_AXES_CUBE_SIZE, 0.000, 1], dtype=np.float64),
    np.array([0.000, 0.000, -CAM_AXES_CUBE_SIZE, 1], dtype=np.float64),
    np.array([CAM_AXES_CUBE_SIZE, 0.000, -CAM_AXES_CUBE_SIZE, 1], dtype=np.float64),
    np.array([CAM_AXES_CUBE_SIZE, CAM_AXES_CUBE_SIZE, -CAM_AXES_CUBE_SIZE, 1], dtype=np.float64),
    np.array([0.000, CAM_AXES_CUBE_SIZE, -CAM_AXES_CUBE_SIZE, 1], dtype=np.float64),
]
CAM_AXES_IDS =  [[0, 1],[1, 2],[2, 3],[3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]
MAP_CONFIG = {
    2: ((0.050, 0.050, 0.000),
        (0.000, 0.050, 0.000),
        (0.000, 0.000, 0.000),
        (0.050, 0.000, 0.000)),
    3: ((0.350, 0.050, 0.000),
        (0.300, 0.050, 0.000),
        (0.300, 0.000, 0.000),
        (0.350, 0.000, 0.000)),
    0: ((0.350, 0.350, 0.000),
        (0.300, 0.350, 0.000),
        (0.300, 0.300, 0.000),
        (0.350, 0.300, 0.000)),
    1: ((0.050, 0.350, 0.000),
        (0.000, 0.350, 0.000),
        (0.000, 0.300, 0.000),
        (0.050, 0.300, 0.000))
}

with open('calibration/param.txt') as f:
    cameraMatrix = eval(f.readline())
    distCoeffs = eval(f.readline())

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

w, h = CAM_RESOLUTION
cap = cv2.VideoCapture(CAM_ID, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
cap.set(cv2.CAP_PROP_FOCUS, 250)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M','J','P','G'))

world_transform = None

if __name__ == "__main__":
    cTw = np.array([])
    while cv2.waitKey(1000//30) != ord("q"):
        ret, frame = cap.read()
        assert ret
        img = cv2.undistort(frame, cameraMatrix, distCoeffs)
        out = img.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = detector.detectMarkers(gray)

        assert corners is not None and ids is not None

        map_markers = {}
        target_markers = {}
        for marker in range(len(ids)):
            idx = int(ids[marker][0])
            cornersx = corners[marker]
            if idx in MAP_CONFIG:
                map_markers[MAP_CONFIG[idx]] = cornersx
            if idx in TARGET_ARUCO_IDS:
                target_markers[idx] = cornersx

        if world_transform is None or len(cTw) == 0:
            objectPoints = []
            imagePoints = []
            for world_position in map_markers:
                objectPoints += list(world_position)
                imagePoints += list(map_markers[world_position][0])
            objectPoints = np.array(objectPoints, dtype=np.float64)
            imagePoints = np.array(imagePoints, dtype=np.float64)

            ret, rvec, tvec = cv2.solvePnP(objectPoints, imagePoints,
                                           cameraMatrix, np.array([]),
                                           useExtrinsicGuess=False,
                                           flags=cv2.SOLVEPNP_IPPE)
            assert ret
            world_transform = (rvec, tvec)

            cTw = re3d.getCTW(rvec, tvec)
            assert np.linalg.det(cTw) != 0
        else:
            out = cv2.aruco.drawDetectedMarkers(out,
                                                corners,
                                                ids,
                                                (255, 0, 255))
            for idx in target_markers:
                cornersx = target_markers[idx]
                ret, rvecx, tvecx = re3d.estimatePoseSingleMarkers(cornersx, TARGET_ARUCO_SIZE, cameraMatrix, np.array([]))
                assert ret
                corners_3d_pos = re3d.get3DMarkerCorners(TARGET_ARUCO_SIZE, rvecx, tvecx)
                icTw = np.linalg.inv(cTw)
                iZ = np.eye(4,  dtype=np.float64)
                iZ[2][2] = -1
                corner_3d_point = [np.dot(iZ, np.dot(icTw, np.array([*corner_3d_pos, 1], dtype=np.float32))) for corner_3d_pos in corners_3d_pos]
                print(idx, corner_3d_point)
                out = cv2.putText(out, f"{[float(round(point, 3)) for point in corner_3d_point[0]][:3]}", np.array(cornersx[0][0]).astype(np.int16), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

            rvec, tvec = world_transform
            camera_axes_points = np.array([np.dot(cTw, point)[:3] for point in CAM_AXES_POINTS], dtype=np.float64)
            out_points, out_jacobian_mat = cv2.projectPoints(camera_axes_points, np.zeros((3, 1), np.float32), np.zeros((3, 1), np.float32), cameraMatrix, np.array([], dtype=np.float64))
            out_points = np.reshape(out_points, (8, 2)).astype(np.int16)
            for fp, tp in CAM_AXES_IDS:
                out = cv2.line(out, out_points[fp], out_points[tp], (200, 0, 200), 2)
            out = cv2.drawFrameAxes(out, cameraMatrix, np.array([]), rvec, tvec, 0.1, 5)


        cv2.imshow("Output Display", camutil.dev_image(out, 2))
        cv2.imwrite("out.jpg", out)
        cv2.imwrite("img.jpg", img)
