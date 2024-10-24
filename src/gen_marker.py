import numpy as np
import cv2
import cv2.aruco as aruco

def gen_arcomarker(marker_id):
    arco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
    marker_size = 200

    marker_image = np.zeros((marker_size, marker_size), dtype=np.uint8)
    marker_image = aruco.generateImageMarker(arco_dict, marker_id, marker_size)

    output_file = f"aruco_marker_{marker_id}.png"
    cv2.imwrite(output_file, marker_image)

    print(f"ArUco marker generated: {output_file}")


def main():
    for marker_id in range(1, 5):
        gen_arcomarker(marker_id)

if __name__ == "__main__":
    main()