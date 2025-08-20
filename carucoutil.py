import cv2
import numpy as np


def conbits(bitdata: str, add_significant: bool = False):
    bytedata = []
    nbytes = int(np.ceil(len(bitdata)/8))
    for i in range(nbytes):
        bytedata.append([])
    for i in range(len(bitdata)):
        bytedata[i//8].append(bitdata[i])
    if add_significant:
        for i in range(len(bitdata), nbytes*8):
            bytedata[i//8].append("0")
    for i in range(nbytes):
        bytedata[i] = int("".join(bytedata[i]), 2)
    return bytedata


def generate_aruco(markerSize: int = 4, data: dict = None, add_significant: bool = False):
    if data is None:
        data = {
            0: [0]*(markerSize*markerSize)
        }

    bdata = {}
    for i in data:
        bdata[i] = []
        for k in range(4):
            rb = conbits(
                "".join(map(str, np.rot90(np.array(data[i]).reshape(markerSize, markerSize), k).ravel())),
                add_significant=add_significant
            )
            bdata[i].append(rb)

    nbytes = int(np.ceil(markerSize*markerSize/8))
    max_i = max(data.keys())

    mat = np.zeros((max_i + 1, nbytes, 4), dtype="uint8")

    aruco_dict = cv2.aruco.Dictionary(mat, markerSize)

    for i in data:
        for k in range(4):
            for j in range(nbytes):
                aruco_dict.bytesList[i].ravel()[k*nbytes + j] = bdata[i][k][j]
    return aruco_dict

def save_aruco_dict(name: str, aruco_dict: cv2.aruco.Dictionary):
    fs = cv2.FileStorage(f"{name}.yaml", cv2.FILE_STORAGE_WRITE)
    aruco_dict.writeDictionary(fs, f"ArUcoDict_{name}")
    fs.release()


def load_aruco_dict(name: str):
    fs = cv2.FileStorage(f"{name}.yaml", cv2.FILE_STORAGE_READ)
    fn = fs.getNode(f"ArUcoDict_{name}")
    data = cv2.aruco.Dictionary()
    assert data.readDictionary(fn)
    fs.release()
    return data


if __name__ == "__main__":
    DICT_3X3_RX_EURO = {
        0: [
            1, 1, 1,
            0, 1, 0,
            0, 1, 0
        ],
        1: [
            1, 0, 1,
            1, 1, 1,
            0, 1, 0
        ],
        2: [
            1, 1, 1,
            0, 0, 0,
            1, 0, 1
        ],
        3: [
            1, 1, 1,
            1, 1, 1,
            0, 1, 0
        ],
        4: [
            1, 0, 0,
            1, 1, 1,
            1, 1, 0
        ],
        5: [
            1, 1, 1,
            1, 0, 0,
            1, 0, 0
        ],
        6: [
            1, 1, 1,
            1, 0, 1,
            0, 0, 0
        ],
        7: [
            1, 0, 0,
            0, 0, 1,
            0, 1, 1
        ]
    }
    save_aruco_dict("DICT_3X3_RX_EURO", generate_aruco(3, DICT_3X3_RX_EURO))
