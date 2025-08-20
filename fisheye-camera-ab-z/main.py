import cv2
import numpy as np
import numpy.typing as npt
import camutil
import re3d

CAM_ID = 1
TARGET_ARUCO_IDS = [80]
TARGET_ARUCO_SIZE = 0.06
TARGET_ARUCO_Z = -0.208
FISHEYE_KNEW_C = 1.25

CAM_RESOLUTION = (3840, 2160)
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

def pgram_center_xy(pts: npt.ArrayLike) -> npt.ArrayLike:
    ptx = float(sum([pts[i][0] for i in range(4)])) / 4
    pty = float(sum([pts[i][1] for i in range(4)])) / 4
    return np.array([ptx, pty], dtype=np.float32)

def pgram_center_xyz(pts: npt.ArrayLike) -> npt.ArrayLike:
    ptx = float(sum([pts[i][0] for i in range(4)])) / 4
    pty = float(sum([pts[i][1] for i in range(4)])) / 4
    ptz = float(sum([pts[i][2] for i in range(4)])) / 4
    return np.array([ptx, pty, ptz], dtype=np.float32)

with open('calibration/param.txt') as f:
    K = eval(f.readline())
    D = eval(f.readline())
    Knew = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, CAM_RESOLUTION, np.eye(3), new_size=CAM_RESOLUTION, fov_scale=FISHEYE_KNEW_C)



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
    hwk = 0
    while hwk != ord("q"):
        ret, frame = cap.read()
        assert ret

        img = cv2.fisheye.undistortImage(frame, K, D=D, Knew=Knew)
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
                                           Knew, np.array([]),
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
                ret, rvecx, tvecx = re3d.estimatePoseSingleMarkers(cornersx, TARGET_ARUCO_SIZE, Knew, np.array([]),)
                assert ret
                corx = (cornersx[0][0][0]+cornersx[0][1][0]+cornersx[0][2][0]+cornersx[0][3][0])/4
                cory = (cornersx[0][0][1]+cornersx[0][1][1]+cornersx[0][2][1]+cornersx[0][3][1])/4
                #print("2d", corx, cory, w, h)
                #print("PnP", tvecx.reshape(3).tolist())
                corners_3d_pos = re3d.get3DMarkerCorners(TARGET_ARUCO_SIZE, rvecx, tvecx)
                icTw = np.linalg.inv(cTw)

                corner_3d_point = [np.dot(icTw, np.array([*corner_3d_pos, 1], dtype=np.float32)) for corner_3d_pos in corners_3d_pos]
                center_3d_point = pgram_center_xyz(corner_3d_point)

                corner_3d_point_zfix = re3d.getFixedZWPos([corx, cory], TARGET_ARUCO_Z, Knew, icTw)

                out = cv2.putText(out, f"SolvePnP: {[float(round(float(point), 3)) for point in center_3d_point]}",
                                  np.array(cornersx[0][0]).astype(np.int16), cv2.FONT_HERSHEY_SIMPLEX, 3, (32, 255, 32), 8)
                out = cv2.putText(out, f"Fixed Z: {[float(round(float(point), 3)) for point in corner_3d_point_zfix]}",
                                  (int(cornersx[0][0][0]),int(cornersx[0][0][1]+100)), cv2.FONT_HERSHEY_SIMPLEX, 3, (32, 255, 32),
                                  8)


            rvec, tvec = world_transform
            camera_axes_points = np.array([np.dot(cTw, point)[:3] for point in CAM_AXES_POINTS], dtype=np.float64)
            out_points, out_jacobian_mat = cv2.projectPoints(camera_axes_points, np.zeros((3, 1), np.float32), np.zeros((3, 1), np.float32), Knew, np.array([], dtype=np.float64))
            out_points = np.reshape(out_points, (8, 2)).astype(np.int16)
            for fp, tp in CAM_AXES_IDS:
                out = cv2.line(out, out_points[fp], out_points[tp], (200, 0, 200), 4)
            out = cv2.drawFrameAxes(out, Knew, np.array([]), rvec, tvec, 0.1, 10)

        out_points, out_jacobian_mat = cv2.projectPoints(np.array([[0.0,0.0,0.5]]), np.zeros((3, 1), np.float32),
                                                         np.zeros((3, 1), np.float32), Knew,
                                                         np.array([], dtype=np.float64))
        out = cv2.circle(out, out_points[0][0].astype(np.int16).tolist(), 16, (0, 0, 255), -1)
        cv2.imshow("Output Display", camutil.dev_image(out, 5))
        hwk = cv2.waitKey(1000 // 30)
        if hwk == ord("s"):
            cv2.imwrite("../out_images/0.150x0.150_out.jpg", out)
            cv2.imwrite("../out_images/0.150x0.150_img.jpg", img)
