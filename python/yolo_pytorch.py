import torch
import onnx
import onnxruntime
import numpy as np
import cv2

# Verificar el modelo ONNX
onnx_model_path = "yolov5-opencv-cpp-python-main/config_files/yolov5n.onnx"
onnx_model = onnx.load(onnx_model_path)
onnx.checker.check_model(onnx_model)
print("El modelo ONNX es válido.")

# Crear una sesión de inferencia con ONNX Runtime (con GPU)
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
session = onnxruntime.InferenceSession(onnx_model_path, providers=providers)

# Cargar las clases
with open("yolov5-opencv-cpp-python-main/config_files/classes.txt", "r") as f:
    class_list = [cname.strip() for cname in f.readlines()]

# Procesar un cuadro
def process_frame(frame):
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (640, 640), swapRB=True)
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: blob})[0]
    return outputs

# Dibujar las detecciones en el cuadro
def draw_boxes(frame, outputs, conf_threshold=0.4, nms_threshold=0.4):
    h, w, _ = frame.shape
    boxes = []
    confidences = []
    class_ids = []

    for output in outputs[0]:
        # Convertir las salidas en formato adecuado
        scores = output[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        if confidence > conf_threshold:
            center_x = int(output[0] * w)
            center_y = int(output[1] * h)
            width = int(output[2] * w)
            height = int(output[3] * h)
            x = int(center_x - width / 2)
            y = int(center_y - height / 2)

            boxes.append([x, y, width, height])
            confidences.append(float(confidence))
            class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            label = f"{class_list[class_ids[i]]} ({confidences[i]:.2f})"
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# Capturar imágenes desde la cámara
cap = cv2.VideoCapture(3)

while True:
    ret, frame = cap.read()
    if not ret:
        print("No se pudo leer el cuadro.")
        break

    # Inferencia
    outputs = process_frame(frame)
    cv2.imshow("Detección ONNX Runtime", frame)

    # Dibujar las cajas de detección
    draw_boxes(frame, outputs)

    # Mostrar el cuadro
    # cv2.imshow("Detección ONNX Runtime", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
