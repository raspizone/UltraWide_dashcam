import cv2
import numpy as np


# Función para calcular la homografía utilizando SIFT
def calculate_homography(img1, img2):
    # Convertir las imágenes a escala de grises
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Detectar características usando SIFT
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    # Emparejar características entre las dos imágenes
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    # Extraer los puntos de las características emparejadas
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Estimar la homografía
    H, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
    return H

# Función para crear una panorámica en tiempo real
def create_real_time_panorama(video_path1, video_path2):
    cap1 = cv2.VideoCapture(video_path1)
    cap2 = cv2.VideoCapture(video_path2)

    # Leer el primer frame de cada video para calcular la homografía
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    if not ret1 or not ret2:
        print("Error al leer los videos.")
        return

    # Calcular la homografía
    H = calculate_homography(frame1, frame2)

    # Dimensiones de los frames
    h1, w1, _ = frame1.shape
    h2, w2, _ = frame2.shape

    # Crear una ventana para mostrar la panorámica
    panorama_width = w1 + w2
    panorama_height = max(h1, h2)
    cv2.namedWindow("Panorámica en Tiempo Real", cv2.WINDOW_NORMAL)

    while cap1.isOpened() and cap2.isOpened():
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        if not ret1 or not ret2:
            break

        # Aplicar la homografía para alinear los frames
        panorama = cv2.warpPerspective(frame2, H, (panorama_width, panorama_height))
        panorama[0:h1, 0:w1] = frame1

        # Mostrar la panorámica en la misma ventana
        cv2.imshow("Panorámica en Tiempo Real", panorama)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()

# Rutas de los videos
video_path1 = "/media/sf_sharedVBOX/yolov5-opencv-cpp-python-main/python/cap1.mkv"
video_path2 = "/media/sf_sharedVBOX/yolov5-opencv-cpp-python-main/python/cap2.mkv"

# Crear la panorámica en tiempo real
create_real_time_panorama(video_path1, video_path2)