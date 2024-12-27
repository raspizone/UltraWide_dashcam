import cv2

def resize_without_cropping(frame, target_width, target_height):
    h, w, _ = frame.shape

    # Mantener la relación de aspecto de la cámara, redimensionando sin recorte
    aspect_ratio = w / h
    new_width = target_width
    new_height = int(new_width / aspect_ratio)

    # Si el nuevo alto excede la altura objetivo, ajustamos la altura
    if new_height > target_height:
        new_height = target_height
        new_width = int(new_height * aspect_ratio)

    # Redimensionar sin distorsión
    resized_frame = cv2.resize(frame, (new_width, new_height))
    
    # Crear un fondo negro para ajustar la imagen a las dimensiones de la salida
    top = (target_height - new_height) // 2
    bottom = target_height - new_height - top
    left = (target_width - new_width) // 2
    right = target_width - new_width - left

    # Poner la imagen centrada sobre un fondo negro
    final_frame = cv2.copyMakeBorder(resized_frame, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    
    return final_frame

# Rutas de los videos
video_paths = ["video1.mp4", "video2.mp4", "video3.mp4"]  # Reemplaza con tus videos
captures = [cv2.VideoCapture(path) for path in video_paths]

# Resolución objetivo
final_width = 1280  # Ancho total del video combinado
final_height = 400  # Altura del video combinado
individual_width = final_width // len(video_paths)  # Ancho de cada video individual (426 píxeles por cámara)

while True:
    frames = []
    for capture in captures:
        ret, frame = capture.read()

        if ret:
            # Redimensionar cada frame sin recorte y ajustarlo al tamaño objetivo
            resized_frame = resize_without_cropping(frame, individual_width, final_height)
            frames.append(resized_frame)
        else:
            # Si un video termina, reiniciarlo
            capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

    if len(frames) == len(captures):
        # Combinar los frames horizontalmente
        combined_frame = cv2.hconcat(frames)

        # Mostrar el frame combinado
        cv2.imshow("Combined Videos", combined_frame)

    # Salir al presionar 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
for capture in captures:
    capture.release()
cv2.destroyAllWindows()
