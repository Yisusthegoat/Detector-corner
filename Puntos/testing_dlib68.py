import cv2
import mediapipe as mp
import dlib
import imutils

# Inicializar la captura de la cámara
cap = cv2.VideoCapture(0)

# Detector facial
face_detector = dlib.get_frontal_face_detector()
# Predictor de 68 puntos de referencia
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Inicializar el detector de manos de MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = imutils.resize(frame, width=1080)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    coordinates_bboxes = face_detector(gray, 1)
    
    for c in coordinates_bboxes:
        x_ini, y_ini, x_fin, y_fin = c.left(), c.top(), c.right(), c.bottom()
        cv2.rectangle(frame, (x_ini, y_ini), (x_fin, y_fin), (0, 255, 0), 1)

        shape = predictor(gray, c)
        for i in range(0, 68):
            x, y = shape.part(i).x, shape.part(i).y
            cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)
            cv2.putText(frame, str(i + 1), (x, y - 5), 1, 0.8, (0, 255, 255), 1)

    # Detección de manos
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for idx, landmark in enumerate(hand_landmarks.landmark):
                cx, cy = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                cv2.circle(frame, (cx, cy), 5, (255, 0, 255), -1)

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # Verificar la tecla Esc
        break

cap.release()
cv2.destroyAllWindows()
