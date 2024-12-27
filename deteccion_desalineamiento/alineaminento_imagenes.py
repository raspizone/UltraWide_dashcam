

import cv2
import numpy as np
from sklearn.cluster import KMeans

# Función para realizar K-means y obtener la imagen filtrada con 2 áreas bien diferenciadas
def obtener_imagen_filtrada(img):
    # Convertir la imagen a un array de 2 dimensiones
    datos = img.reshape((-1, 3))

    # Aplicar K-means con 2 clusters
    kmeans = KMeans(n_clusters=2, random_state=0).fit(datos)
    etiquetas = kmeans.labels_
    centros = kmeans.cluster_centers_

    # Crear una imagen filtrada con los centros de los clusters
    img_filtrada = centros[etiquetas].reshape(img.shape).astype(np.uint8)

    return img_filtrada

# Función para encontrar las líneas rectas del rectángulo
def encontrar_rectangulo(mascara):
    # Encontrar contornos en la máscara
    contornos, _ = cv2.findContours(mascara.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Asumimos que el contorno más grande es el rectángulo negro
    contorno_mayor = max(contornos, key=cv2.contourArea)
    
    # Obtener el rectángulo delimitador
    x, y, ancho, alto = cv2.boundingRect(contorno_mayor)

    return x, y, ancho, alto


# Función para alinear dos imágenes utilizando SIFT
def align_images_sift(img1, img2):
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

    # Estimar la homografía y alinear las imágenes
    H, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
    h, w, _ = img1.shape
    aligned_img = cv2.warpPerspective(img2, H, (w, h))

    return aligned_img

# Cargar las imágenes de prueba
img1 = cv2.imread("/media/sf_sharedVBOX/yolov5-opencv-cpp-python-main/python/cap1.png")
img2 = cv2.imread("/media/sf_sharedVBOX/yolov5-opencv-cpp-python-main/python/cap2.png")

# Verificar que las imágenes se hayan cargado correctamente
if img1 is None or img2 is None:
    print("Error al cargar las imágenes. Verifica las rutas de los archivos.")
else:
    # Alinear las imágenes utilizando SIFT
    aligned_img_sift = align_images_sift(img1, img2)

img = aligned_img_sift

# Verificar que la imagen se haya cargado correctamente
if img is None:
    print("Error al cargar la imagen. Verifica la ruta del archivo.")
else:
    # Obtener la imagen filtrada utilizando K-means
    img_filtrada = obtener_imagen_filtrada(img)

    # Convertir la imagen filtrada a escala de grises
    img_filtrada_gris = cv2.cvtColor(img_filtrada, cv2.COLOR_BGR2GRAY)

    # Aplicar umbral para obtener una máscara binaria
    _, mascara_binaria = cv2.threshold(img_filtrada_gris, 127, 255, cv2.THRESH_BINARY)

    # Encontrar las líneas rectas del rectángulo negro
    x, y, ancho, alto = encontrar_rectangulo(mascara_binaria)

    # Dibujar el rectángulo en la imagen original
    img_rectangulo = img.copy()
    cv2.rectangle(img_rectangulo, (x, y), (x + ancho, y + alto), (0, 255, 0), 3)

  
    print(f"Ancho deslineado = {ancho}")
    

    # Mostrar las imágenes originales y la imagen alineada
    cv2.imshow("Imagen 1", img1)
    cv2.imshow("Imagen 2", img2)
    alineada = cv2.hconcat([img1,img2])
    cv2.imshow("Imagen alineada", alineada)
    cv2.imwrite("diferencia.jpg", aligned_img_sift)
    cv2.waitKey(0)
    cv2.destroyAllWindows()