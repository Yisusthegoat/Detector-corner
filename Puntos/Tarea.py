import cv2
import numpy as np
import face_recognition

# Cargar las imágenes de los rostros conocidos y sus nombres
known_face_encodings = []
known_face_names = []

def load_known_face(image_path, name):
    image = face_recognition.load_image_file(image_path)
    encoding = face_recognition.face_encodings(image)[0]
    known_face_encodings.append(encoding)
    known_face_names.append(name)

load_known_face('known_faces/Jesus_Antonio.jpeg', 'Ñahuinripa Zamudio Jesus Antonio')
#load_known_face('known_faces/favio_samir.jpeg', 'Castillo Lavado Favio Samir')
load_known_face('known_faces/Deivid_Samaniego.jpeg', 'Samaniego Quispe Deivid')
#load_known_face('known_faces/henry_hans.jpeg', 'Huayta Hinojosa Henry Hans')

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

    # Redimensionar el cuadro a 1/4 tamaño para un procesamiento más rápido
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Encontrar todas las ubicaciones y encodings de caras en el cuadro actual de video
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Desconocido"

        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        face_names.append(name)

    # Mostrar los resultados
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Dibujar un cuadro alrededor de la cara
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Dibujar la etiqueta con el nombre debajo de la cara
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Mostrar el cuadro de video
    cv2.imshow('Frame', frame)

    # Esperar 1ms entre cuadros para mantener la velocidad original del video
    if cv2.waitKey(1) & 0xFF == 27:  # Presionar 'Esc' para salir
        break

# Liberar la cámara y cerrar todas las ventanas
cap.release()
cv2.destroyAllWindows()
