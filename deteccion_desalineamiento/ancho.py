

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

# Cargar la imagen de prueba
img = cv2.imread("/media/sf_sharedVBOX/yolov5-opencv-cpp-python-main/python/diferencia.jpg")


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
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
