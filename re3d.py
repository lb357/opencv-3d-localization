"""re3d / 2025 Leonid Briskindov"""
import cv2
import numpy as np
import numpy.typing as npt


def getCTW(rvec: cv2.typing.MatLike, tvec: cv2.typing.MatLike) -> npt.ArrayLike:
    """
    Функция преобразования rvec (векора поворота) и tvec (вектора сдвига) в cTw (матрицу перехода)
    """
    
    rot_mat, jacobian_mat = cv2.Rodrigues(rvec)
    mat = np.array([
        [rot_mat[0][0], rot_mat[0][1], rot_mat[0][2], tvec[0][0]],
        [rot_mat[1][0], rot_mat[1][1], rot_mat[1][2], tvec[1][0]],
        [rot_mat[2][0], rot_mat[2][1], rot_mat[2][2], tvec[2][0]],
        [0, 0, 0, 1]
    ])
    return mat


def estimatePoseSingleMarkers(marker_points: cv2.typing.MatLike,
                              marker_size: float,
                              cameraMatrix: cv2.typing.MatLike,
                              distCoeffs: cv2.typing.MatLike,
                              useEPNP: bool = False) -> tuple[bool, cv2.typing.MatLike, cv2.typing.MatLike]:
    """
    Функция по образу устаревшей cv2.aruco.estimatePoseSingleMarkers с немного отличающимися аргументами:
    ! marker_points - позиции (px) вершин (углов) маркера на изображении
    ! marker_size - размер маркера в мировой системе координат (реальный размер, например в метрах)
    ! cameraMatrix - внутренняя матрица камеры
    ! distCoeffs - коэффициенты дисторсии камеры (на неискажённых изображениях ожидается пустой массив)
    ! useEPNP - использовать EPNP метод решения задачи Perspective-n-Point вместо IPPE (SQUARE)
    """
    marker_world_points = np.array([[-marker_size / 2, marker_size / 2, 0],
                                    [marker_size / 2, marker_size / 2, 0],
                                    [marker_size / 2, -marker_size / 2, 0],
                                    [-marker_size / 2, -marker_size / 2, 0]], dtype=np.float32)
    if useEPNP:
        return cv2.solvePnP(marker_world_points, marker_points, cameraMatrix, distCoeffs, flags=cv2.SOLVEPNP_EPNP)
    else:
        return cv2.solvePnP(marker_world_points, marker_points, cameraMatrix, distCoeffs, flags=cv2.SOLVEPNP_IPPE_SQUARE)


def get3D4Points(points: list, rvec: cv2.typing.MatLike, tvec: cv2.typing.MatLike) -> npt.ArrayLike:
    """
    Функция применения преобразования, описываемого векторами rvec и tvec, к четырём точкам (из входного массива points)
    """
    mat = getCTW(rvec, tvec)
    camera_points = np.array([
        np.dot(mat, points[0]),
        np.dot(mat, points[1]),
        np.dot(mat, points[2]),
        np.dot(mat, points[3])
    ])
    return camera_points[:, :-1]


def get3DMarkerCorners(marker_size: float, rvec: cv2.typing.MatLike, tvec: cv2.typing.MatLike) -> npt.ArrayLike:
    """
    Функция применения преобразования, описываемого векторами rvec и tvec, к вершинам (углам) ArUco маркера c размером в мировой системе координат (реальным размером) = marker_size (например в метрах)
    """
    marker_world_points = np.array([[-marker_size / 2, marker_size / 2, 0, 1],
                                    [marker_size / 2, marker_size / 2, 0, 1],
                                    [marker_size / 2, -marker_size / 2, 0, 1],
                                    [-marker_size / 2, -marker_size / 2, 0, 1]], dtype=np.float32)
    return get3D4Points(marker_world_points, rvec, tvec)


def getKnew(K: cv2.typing.MatLike, c: float) -> npt.ArrayLike:
    """
    Функция пропорционального изменения фокусного расстояния для нахождения новой внутренней матрицы (Knew)
    Аналогичного результата можно добиться функцией cv2.fisheye.estimateNewCameraMatrixForUndistortRectify, хотя изначально она создана для других целей
    """
    Knew = K.copy()
    Knew[(0, 1), (0, 1)] = c * Knew[(0, 1), (0, 1)]
    return Knew


def getFixedZWPosAll(src: [tuple, list, npt.ArrayLike], Zw: float, cameraMatrix: cv2.typing.MatLike, wTc: npt.ArrayLike) -> (npt.ArrayLike, npt.ArrayLike):
    """
    Функция нахождения позиции точки в мировой системе координат и системе координат камеры с известной координатой Z в мировой системе координат и матрицей перехода из системы координат камеры в мировую систему коодринат:
    ! src - позиция (px) искомой точки на изображении
    ! Zw - координата Z точки в мировой системе координат
    ! cameraMatrix - внутренняя матрица камеры
    ! wTc - матрица перехода из системы координат камеры в мировую систему коодринат (обратная матрица к cTw)
    """
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

    return np.array([Xw, Yw, Zw], dtype=np.float32), np.array([Xc, Yc, Zc], dtype=np.float32)


def getFixedZWPos(src: [tuple, list, npt.ArrayLike], Zw: float, cameraMatrix: cv2.typing.MatLike, wTc: npt.ArrayLike) -> npt.ArrayLike:
    """
    Функция аналогична getFixedZWPosAll, но возвращает только позицию в мировой системе координат
    """
    return getFixedZWPosAll(src, Zw, cameraMatrix, wTc)[0]


def positionMarker(
        marker_corners: cv2.typing.MatLike, marker_size: float, cameraMatrix: cv2.typing.MatLike,
        distCoeffs: cv2.typing.MatLike = np.array([],dtype=np.float32)
    ) -> ([npt.ArrayLike, npt.ArrayLike], [npt.ArrayLike, npt.ArrayLike]):
    """
    Функция нахождения позиции и углов Эйлера ArUco маркера в системе координат камеры.
    ! marker_corners - позиции (px) вершин (углов) маркера на изображении
    ! marker_size - размер маркера в мировой системе координат (реальный размер, например в метрах)
    ! cameraMatrix - внутренняя матрица камеры
    ! distCoeffs - коэффициенты дисторсии камеры (на неискажённых изображениях ожидается пустой массив)
    """
    marker_points = np.array(
        [[-marker_size / 2, marker_size / 2, 0], [marker_size / 2, marker_size / 2, 0],
         [marker_size / 2, -marker_size / 2, 0], [-marker_size / 2, -marker_size / 2, 0]], dtype=np.float32) # Определение объектной модели ArUco маркера
                                                                                                             # Вершины (углы) ArUco маркера описывают квадрат с длиной стороны marker_size и с центром в начале системы координат
    ret, rvec, tvec = cv2.solvePnP(
        marker_points, marker_corners, cameraMatrix, distCoeffs, flags=cv2.SOLVEPNP_IPPE_SQUARE
    )                                                                                                        # Получение вектора поворота и сдвига rvec и tvec, описывающих преобразование из мировой системы координат в систему
                                                                                                             # координат камеры

    assert ret                                                                                               # Проверка на успешность выполнения cv2.solvePnP на предыдущем шаге

    x, y, z = tvec.ravel()                                                                                   # Разложение tvec на x, y, z для удобства формирования вывода (tvec / t -> x, y, z)
    rot_mat, jacobian_mat = cv2.Rodrigues(rvec)                                                              # Нахождение матрицы поворота из вектора поворота (rvec -> R)
    ax = np.arctan2(rot_mat[2][1], rot_mat[2][2])                                                            # Нахождение угла Эйлера из матрицы поворота относительно OX
    ay = np.arctan2(-1 * rvec[2][0], np.sqrt((rot_mat[2][1]) ** 2 + (rot_mat[2][2]) ** 2))                   # Нахождение угла Эйлера из матрицы поворота относительно OY
    az = np.arctan2(rot_mat[1][0], rot_mat[0][0])                                                            # Нахождение угла Эйлера из матрицы поворота относительно OZ
    return np.array([[x, y, z], [ax, ay, az]], dtype=np.float32), np.array([rvec, tvec], dtype=np.float32)   # Возврат функции ([Координаты, Углы], [rvec, tvec])
