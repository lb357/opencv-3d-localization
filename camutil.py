import cv2
import numpy as np
import numpy.typing as npt


def debug_image(image: cv2.typing.MatLike, debug_mode: bool = True) -> None:
    if debug_mode:
        debug_window_name = "CAMUTIL IMAGE DEBUG"
        cv2.namedWindow(debug_window_name)
        while cv2.waitKey(1) != ord("q"):
            cv2.imshow(debug_window_name, image)
        cv2.destroyWindow(debug_window_name)


def dev_image(image: cv2.typing.MatLike, k: float) -> cv2.typing.MatLike:
    return cv2.resize(image.copy(), (image.shape[1]//k, image.shape[0]//k))
