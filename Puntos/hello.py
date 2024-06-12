import cv2
import numpy as np

# Abrir la cámara
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error al abrir la cámara")
    exit()

# Crear una ventana y configurar a pantalla completa
cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
cv2.setWindowProperty('Frame', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Definir la región de interés (ROI)
    height, width, _ = frame.shape
    roi = frame[int(0.1 * height):int(0.9 * height), int(0.1 * width):int(0.9 * width)]

    # Convertir a escala de grises
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # Convertir a float32
    gray = np.float32(gray)
    
    # Detectar esquinas usando el detector de Harris
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)
    
    # Dilatar el resultado para marcar las esquinas
    dst = cv2.dilate(dst, None)
    
    # Crear una imagen en negro para los puntos detectados
    points_frame = np.zeros_like(roi)
    
    # Umbralizar para obtener los puntos importantes
    threshold = 0.005 * dst.max()  # Reducir el umbral para detectar más esquinas
    points_frame[dst > threshold] = [0, 255, 255]  # Esquinas marcadas en color

    # Combinar los dos cuadros
    combined_frame = np.hstack((roi, points_frame))
    
    # Añadir texto a los cuadros
    cv2.putText(combined_frame, 'Original Video', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(combined_frame, 'Detected important corners', (roi.shape[1] + 50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Mostrar el cuadro combinado
    cv2.imshow('Frame', combined_frame)
    
    # Esperar 1ms entre cuadros para mantener la velocidad original del video
    if cv2.waitKey(1) & 0xFF == 27:  # Presionar 'Esc' para salir
        break

# Liberar la cámara y cerrar todas las ventanas
cap.release()
cv2.destroyAllWindows()
