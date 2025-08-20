import cv2
import numpy as np
import re3d
# Импорт библиотек

CAM_RESOLUTION = (1920, 1080) # Разрешение камеры
ARUCO_SIZE = 0.0585           # Размер ArUco маркера около 6 см
                              # (ArUco маркер распечатан, вырезан и замерен линейкой)
CAM_ID = 1                    # ID камеры (при подключении по USB)

with open('calibration/param.txt') as f: 
    cameraMatrix = eval(f.readline())
    distCoeffs = eval(f.readline())
# Загрузка из 'calibration/param.txt' матрицы камеры и коэффициентов дисторсии

w, h = CAM_RESOLUTION # Разбиение разрешения камеры на длину и высоту для удобства
cap = cv2.VideoCapture(CAM_ID, cv2.CAP_DSHOW) # Создание объекта камеры
cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
cap.set(cv2.CAP_PROP_FOCUS, 250)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*'MJPG'))
# Настройка камеры

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
# Загрузка стандартного словаря ArUco маркеров 4x4
parameters = cv2.aruco.DetectorParameters()
# Создание объекта параметров детектора ArUco маркеров
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
# Создание объекта детектора ArUco маркеров

if __name__ == "__main__":
    while cv2.waitKey(1000 // 60) != ord("q"): 
        # Обновление изображения до нажатия на клавишу "q"  
        ret, frame = cap.read()
        assert ret
        # Получение изображения с камеры
        img = cv2.fisheye.undistortImage(frame, cameraMatrix, D=distCoeffs, Knew=cameraMatrix)
        # Удаление искажения с камеры     
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Создание чёрно-белого изображения камеры
        corners, ids, rejected = detector.detectMarkers(gray)
        # Поиск ArUco маркеров
        if ids is not None:
            # Если найден(-ы) ArUco маркер(-ы):
            for marker in range(len(ids)):
                # Выполнить для каждого найденного ArUco маркера
                idx = int(ids[marker][0]) # Код (номер) ArUco маркера
                cornersx = corners[marker] # Вершины (углы) ArUco маркера
                position, mat = re3d.positionMarker(cornersx, ARUCO_SIZE, cameraMatrix)
                # Рассчёт позиции ArUco маркера (смотри re3d.py)
                x, y, z = position[0] # Координаты
                rx, ry, rz = map(np.degrees, position[1]) # Углы
                rvec, tvec = mat # rvec, tvec (для отрисовки осей ArUco маркера)
                img = cv2.drawFrameAxes(img, cameraMatrix, np.array([]), rvec, tvec, 0.1, 5)
                #Отрисовка осей ArUco маркера
                img_pos = np.array(cornersx[0][0]).astype(np.int16)
                #Позиция (px) вершины (угла) ArUco маркера на изображении
                img = cv2.putText(
                    img,
                    f"x:{x:.2f}/y:{y:.2f}/z:{z:.2f}",
                    img_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 2
                )
                # Отрисовка координат ArUco маркера на изображении
                img = cv2.putText(
                    img,
                    f"rx:{rx:.2f}/ry:{ry:.2f}/rz:{rz:.2f}",
                    [img_pos[0], img_pos[1]+20], cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (255, 0, 255), 2
                )
                # Отрисовка углов ArUco маркера на изображении

        cv2.imshow("Display", img) # Вывод на экран изображения
        
    cap.release()
    cv2.destroyAllWindows()
    # Закрытие окон после завершения работы
