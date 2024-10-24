import cv2
import cv2.aruco as aruco
from PIL import Image

import numpy as np

def detect_marker(file_path, output_path):
    # 検出対象の画像を読み込む
    image = imread_japanese(file_path)

    # グレースケールに変換
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # ArUcoマーカーの辞書を選択（例: 6x6, 250種類のマーカー）
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)

    # ArUcoマーカーのパラメータを設定
    parameters = aruco.DetectorParameters()

    # ArUcoマーカーの検出
    corners, ids, rejected_img_points = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    # マーカーが検出された場合
    if ids is not None:
        print(f"{len(ids)}個のArUcoマーカーが検出されました。")

        # 検出されたマーカーを画像上に描画
        result_image = aruco.drawDetectedMarkers(image, corners, ids)

        # 検出された各マーカーのIDと位置情報を出力
        for i, corner in enumerate(corners):
            marker_id = ids[i][0]
            print(f"ID: {marker_id}, コーナー座標: {corner}")

        # OpenCVのBGRをRGBに変換
        image_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)

        pil_image = Image.fromarray(image_rgb)
        pil_image.save(output_path)

        print(f'検出結果が{output_path}に保存されました')


    else:
        print("ArUcoマーカーが見つかりませんでした。")

def imread_japanese(path):
    # ファイルをバイナリモードで読み込み
    with open(path, 'rb') as f:
        binary_data = np.frombuffer(f.read(), dtype=np.uint8)
    # 画像をデコード
    image = cv2.imdecode(binary_data, cv2.IMREAD_COLOR)
    return image

def drawResult(image, corners, ids):
    copy_image = image.copy()

    for i, corner in enumerate(corners):
        cv2.polylines(copy_image, [np.int32(corner)], True, (0, 255, 0), 2)
        cv2.putText(copy_image, str(ids[i][0]), tuple(np.int32(corner[0])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        marker_id = ids[i][0]
        print(f"ID: {marker_id}, コーナー座標: {corner}")

    return copy_image

def main():
    image_path = r"E:\reps\MarkerGen\markers.tif"
    image_path = r"E:\reps\MarkerGen\test.jpg"
    image_out = r"E:\reps\MarkerGen\test_output.tif"
    detect_marker(image_path, image_out)

if __name__ == "__main__":
    main()
