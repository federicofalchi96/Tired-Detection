import cv2
import time
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Disabilita log TensorFlow (0=default, 1=warning, 2=error, 3=solo errori gravi)
import mediapipe as mp
import pyautogui
import winsound

def play_alert_sound():
    winsound.PlaySound("beep.wav", winsound.SND_FILENAME)

# Setup mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                  max_num_faces=1,
                                  refine_landmarks=True,
                                  min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)

# Webcam
cap = cv2.VideoCapture(0)

# Parametri soglie
EYE_CLOSED_THRESHOLD = 0.2
EYE_CLOSED_SECONDS = 2.0
GAZE_DOWN_SECONDS = 3.0
LOOK_DOWN_THRESHOLD = 0.65
MAR_YAWN_THRESHOLD = 0.6
YAWN_DURATION = 1.5
PERCLOS_WINDOW = 30
BLINK_DURATION = 0.2

eye_closed_start = None
gaze_down_start = None
yawn_start = None
last_blink_time = None
blink_count = 0
eye_closure_times = []

LEFT_EYE = [362, 385, 387, 263, 373, 380]

def get_eye_ratio(landmarks, w, h, eye_indices):
    p = [landmarks[i] for i in eye_indices]
    coords = [(int(pnt.x * w), int(pnt.y * h)) for pnt in p]
    vertical = abs(coords[1][1] - coords[5][1]) + abs(coords[2][1] - coords[4][1])
    horizontal = abs(coords[0][0] - coords[3][0])
    return vertical / (2.0 * horizontal)

def get_mar(landmarks, w, h):
    top = landmarks[13]
    bottom = landmarks[14]
    left = landmarks[78]
    right = landmarks[308]
    vertical = abs(top.y - bottom.y) * h
    horizontal = abs(left.x - right.x) * w
    return vertical / horizontal

def update_perclos(current_time, eyes_closed):
    eye_closure_times.append((current_time, eyes_closed))
    while eye_closure_times and current_time - eye_closure_times[0][0] > PERCLOS_WINDOW:
        eye_closure_times.pop(0)
    closed_frames = sum(1 for t, closed in eye_closure_times if closed)
    return closed_frames / len(eye_closure_times) if eye_closure_times else 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    current_time = time.time()

    if results.multi_face_landmarks:
        face = results.multi_face_landmarks[0]
        landmarks = face.landmark
        ratio = get_eye_ratio(landmarks, w, h, LEFT_EYE)
        mar = get_mar(landmarks, w, h)

        eyes_closed = ratio < EYE_CLOSED_THRESHOLD
        perclos = update_perclos(current_time, eyes_closed)

        if eyes_closed:
            if eye_closed_start is None:
                eye_closed_start = current_time
            elif current_time - eye_closed_start > EYE_CLOSED_SECONDS:
                print("😴 Occhi chiusi troppo a lungo! STANCHEZZA rilevata.")
                pyautogui.screenshot("stanchezza_occhi.png")
                play_alert_sound()
                eye_closed_start = None
            if last_blink_time is None:
                last_blink_time = current_time
        else:
            if last_blink_time:
                if current_time - last_blink_time < BLINK_DURATION:
                    blink_count += 1
                last_blink_time = None
            eye_closed_start = None

        if mar > MAR_YAWN_THRESHOLD:
            if yawn_start is None:
                yawn_start = current_time
            elif current_time - yawn_start > YAWN_DURATION:
                print("😮 Sbadiglio rilevato! STANCHEZZA rilevata.")
                pyautogui.screenshot("stanchezza_sbadiglio.png")
                play_alert_sound()
                yawn_start = None
        else:
            yawn_start = None

        eye_center_y = landmarks[LEFT_EYE[0]].y
        if eye_center_y > LOOK_DOWN_THRESHOLD:
            if gaze_down_start is None:
                gaze_down_start = current_time
            elif current_time - gaze_down_start > GAZE_DOWN_SECONDS:
                print("📉 Sguardo basso prolungato! STANCHEZZA rilevata.")
                pyautogui.screenshot("stanchezza_sguardo.png")
                play_alert_sound()
                gaze_down_start = None
        else:
            gaze_down_start = None

        if perclos > 0.7:
            print("📊 PERCLOS > 70% negli ultimi 30 secondi. STANCHEZZA rilevata!")
            pyautogui.screenshot("stanchezza_perclos.png")
            play_alert_sound()

    cv2.imshow("Webcam", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
