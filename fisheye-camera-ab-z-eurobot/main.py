import cv2
import numpy as np
import numpy.typing as npt
import camutil
import carucoutil
import re3d

CAM_ID = 1
TARGET_ARUCO_IDS = [0, 1, 2, 3]
TARGET_ARUCO_SIZE = 0.055
TARGET_ARUCO_Z = -0.45

CAM_RESOLUTION = (1920, 1080)
CAM_AXES_POINTS = [
    np.array([0.000, 0.000, 0.000, 1], dtype=np.float64),
    np.array([3.000, 0.000, 0.000, 1], dtype=np.float64),
    np.array([3.000, 2.000, 0.000, 1], dtype=np.float64),
    np.array([0.000, 2.000, 0.000, 1], dtype=np.float64),
    np.array([0.000, 0.000, TARGET_ARUCO_Z, 1], dtype=np.float64),
    np.array([3.000, 0.000, TARGET_ARUCO_Z, 1], dtype=np.float64),
    np.array([3.000, 2.000, TARGET_ARUCO_Z, 1], dtype=np.float64),
    np.array([0.000, 2.000, TARGET_ARUCO_Z, 1], dtype=np.float64),
]
CAM_AXES_IDS = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]
MAP_CONFIG = {
    23: ((0.650, 0.650, 0.000),
         (0.550, 0.650, 0.000),
         (0.550, 0.550, 0.000),
         (0.650, 0.550, 0.000)),
    22: ((3.000 - 0.550, 0.650, 0.000),
         (3.000 - 0.650, 0.650, 0.000),
         (3.000 - 0.650, 0.550, 0.000),
         (3.000 - 0.550, 0.550, 0.000)),
    20: ((3.000 - 0.550, 2.000 - 0.550, 0.000),
         (3.000 - 0.650, 2.000 - 0.550, 0.000),
         (3.000 - 0.650, 2.000 - 0.650, 0.000),
         (3.000 - 0.550, 2.000 - 0.650, 0.000)),
    21: ((0.650, 2.000 - 0.550, 0.000),
         (0.550, 2.000 - 0.550, 0.000),
         (0.550, 2.000 - 0.650, 0.000),
         (0.650, 2.000 - 0.650, 0.000)),
}



def getFixedZWPos(src, Zw, cameraMatrix, wTc):
    fx, fy = cameraMatrix[0][0], cameraMatrix[1][1]
    cx, cy = cameraMatrix[0][2], cameraMatrix[1][2]

    r11, r12, r13, tx = wTc[0]
    r21, r22, r23, ty = wTc[1]
    r31, r32, r33, tz = wTc[2]

    u, v = src

    Zc = (Zw - tz) / (r31 * (u - cx) / fx + r32 * (v - cy) / fy + r33)

    Xc = (u - cx) * Zc / fx
    Yc = (v - cy) * Zc / fy

    Xw = r11 * Xc + r12 * Yc + r13 * Zc + tx
    Yw = r21 * Xc + r22 * Yc + r23 * Zc + ty
    print("Z", Xc, Yc, Zc, r31, r32, r33, tz)
    print(wTc)
    print(src)
    return np.array([Xw, Yw, Zw], dtype=np.float32)


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


aruco_dict_4x4 = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
parametersm = cv2.aruco.DetectorParameters()
detectorm = cv2.aruco.ArucoDetector(aruco_dict_4x4, parametersm)

aruco_dict_3x3 = carucoutil.load_aruco_dict("DICT_3X3_RX_EURO")
parametersr = cv2.aruco.DetectorParameters()
detectorr = cv2.aruco.ArucoDetector(aruco_dict_3x3, parametersr)

w, h = CAM_RESOLUTION
cap = cv2.VideoCapture(CAM_ID, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
cap.set(cv2.CAP_PROP_FOCUS, 250)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))

world_transform = None

if __name__ == "__main__":
    cTw = np.array([])
    while cv2.waitKey(1000 // 30) != ord("q"):
        ret, frame = cap.read()
        assert ret

        img = cv2.fisheye.undistortImage(frame, K, D=D, Knew=K)

        out = img.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cornersm, idsm, rejectedm = detectorm.detectMarkers(gray)
        cornersr, idsr, rejectedr = detectorr.detectMarkers(gray)


        map_markers = {}
        target_markers = {}
        for marker in range(len(idsm)):
            idx = int(idsm[marker][0])
            cornersx = cornersm[marker]
            if idx in MAP_CONFIG:
                map_markers[MAP_CONFIG[idx]] = cornersx

        if idsr is not None:
            for marker in range(len(idsr)):
                idx = int(idsr[marker][0])
                cornersx = cornersr[marker]
                if idx in TARGET_ARUCO_IDS:
                    target_markers[idx] = cornersx

        cmap_markers = map_markers.copy()
        if world_transform is None or len(cTw) == 0:
            '''
            for fp in cmap_markers:
                for tp in cmap_markers:
                    if not np.array_equal(fp, tp):
                        fpc_m = pgram_center_xyz(fp)
                        fpc_px = pgram_center_xy(map_markers[fp][0])
                        tpc_m = pgram_center_xyz(tp)
                        tpc_px = pgram_center_xy(map_markers[tp][0])
                        pc_m = (float((fpc_m[0] + tpc_m[0]) / 2), float((fpc_m[1]+tpc_m[1]) / 2), float((fpc_m[2] + tpc_m[2]) / 2))
                        pc_px = np.array([[[(fpc_px[0] + tpc_px[0]) / 2, (fpc_px[1] + tpc_px[1]) / 2]]])

                        map_markers[tuple([pc_m])] = pc_px
            '''
            print(map_markers)
            objectPoints = []
            imagePoints = []
            for world_position in map_markers:
                objectPoints += list(world_position)
                imagePoints += list(map_markers[world_position][0])
            objectPoints = np.array(objectPoints, dtype=np.float64)
            imagePoints = np.array(imagePoints, dtype=np.float64)
            #camutil.debug_image(camutil.dev_image(img, 2))
            print(len(objectPoints) // 4)
            ret, rvec, tvec = cv2.solvePnP(objectPoints, imagePoints,
                                           K, np.array([]),
                                           useExtrinsicGuess=False,
                                           #flags=cv2.SOLVEPNP_IPPE)
                                           flags=cv2.SOLVEPNP_EPNP)
            assert ret
            world_transform = (rvec, tvec)

            cTw = re3d.getCTW(rvec, tvec)
            assert np.linalg.det(cTw) != 0
        else:
            out = cv2.aruco.drawDetectedMarkers(out,
                                                cornersm,
                                                idsm,
                                                (255, 0, 255))
            out = cv2.aruco.drawDetectedMarkers(out,
                                                cornersr,
                                                idsr,
                                                (255, 255, 0))
            for idx in target_markers:
                cornersx = target_markers[idx]
                ret, rvecx, tvecx = re3d.estimatePoseSingleMarkers(cornersx, TARGET_ARUCO_SIZE, K, np.array([]), True)
                assert ret
                corx = (cornersx[0][0][0] + cornersx[0][1][0] + cornersx[0][2][0] + cornersx[0][3][0]) / 4
                cory = (cornersx[0][0][1] + cornersx[0][1][1] + cornersx[0][2][1] + cornersx[0][3][1]) / 4
                print("2d", corx, cory, w, h)
                print("PnP", tvecx.reshape(3).tolist())
                corners_3d_pos = re3d.get3DMarkerCorners(TARGET_ARUCO_SIZE, rvecx, tvecx)
                icTw = np.linalg.inv(cTw)

                corner_3d_point = [np.dot(icTw, np.array([*corner_3d_pos, 1], dtype=np.float32)) for corner_3d_pos in
                                   corners_3d_pos]
                center_3d_point = pgram_center_xyz(corner_3d_point)

                corner_3d_point_zfix = getFixedZWPos([corx, cory], TARGET_ARUCO_Z, K, icTw)
                print(idx, corner_3d_point, corner_3d_point_zfix)
                out = cv2.putText(out, f"SolvePnP: {[float(round(float(point), 3)) for point in center_3d_point]}",
                                  np.array(cornersx[0][0]).astype(np.int16), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255),
                                  4)
                out = cv2.putText(out, f"Fixed Z: {[float(round(float(point), 3)) for point in corner_3d_point_zfix]}",
                                  (int(cornersx[0][0][0]), int(cornersx[0][0][1] + 75)), cv2.FONT_HERSHEY_SIMPLEX, 2,
                                  (255, 0, 255),
                                  4)

            rvec, tvec = world_transform
            camera_axes_points = np.array([np.dot(cTw, point)[:3] for point in CAM_AXES_POINTS], dtype=np.float64)
            out_points, out_jacobian_mat = cv2.projectPoints(camera_axes_points, np.zeros((3, 1), np.float32),
                                                             np.zeros((3, 1), np.float32), K,
                                                             np.array([], dtype=np.float64))
            out_points = np.reshape(out_points, (8, 2)).astype(np.int16)
            for fp, tp in CAM_AXES_IDS:
                out = cv2.line(out, out_points[fp], out_points[tp], (200, 0, 200), 4)
            out = cv2.drawFrameAxes(out, K, np.array([]), rvec, tvec, 0.1, 10)

        out_points, out_jacobian_mat = cv2.projectPoints(np.array([[0.0, 0.0, 0.5]]), np.zeros((3, 1), np.float32),
                                                         np.zeros((3, 1), np.float32), K,
                                                         np.array([], dtype=np.float64))
        out = cv2.circle(out, out_points[0][0].astype(np.int16).tolist(), 16, (0, 0, 255), -1)
        cv2.imshow("Output Display", camutil.dev_image(out, 2))
        cv2.imwrite("out.jpg", out)
        cv2.imwrite("img.jpg", img)
