import cv2
import os
import numpy as np
import mediapipe as mp
import math

# Cetak direktori kerja saat ini untuk memastikan itu benar
print("Direktori Kerja Saat Ini:", os.getcwd())

# Gunakan jalur absolut untuk memuat model
current_dir = os.path.dirname(os.path.abspath(__file__))
age_prototxt = os.path.join(current_dir, "age_deploy.prototxt")
age_caffemodel = os.path.join(current_dir, "age_net.caffemodel")
gender_prototxt = os.path.join(current_dir, "gender_deploy.prototxt")
gender_caffemodel = os.path.join(current_dir, "gender_net.caffemodel")

# Muat model pralatih untuk deteksi usia dan jenis kelamin
age_net = cv2.dnn.readNetFromCaffe(age_prototxt, age_caffemodel)
gender_net = cv2.dnn.readNetFromCaffe(gender_prototxt, gender_caffemodel)

# Nilai mean untuk model
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

# Daftar usia dan jenis kelamin
age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
gender_list = ['Laki-laki', 'Perempuan']

# Muat Haar cascade untuk deteksi wajah
facereg = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Inisialisasi MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Mulai pengambilan video
cam = cv2.VideoCapture(0)

# Muat gambar objek
object_img = cv2.imread(os.path.join(current_dir, 'pngegg.png'), cv2.IMREAD_UNCHANGED)  # Pastikan menggunakan gambar PNG dengan alpha channel

# Inisialisasi variabel untuk memindahkan objek
object_center = None
pinching = False
tapped = False

# Simpan posisi sebelumnya untuk interpolasi
prev_object_center = None

def pre_process(face_img):
    # Lakukan resizing untuk konsistensi ukuran
    face_img = cv2.resize(face_img, (227, 227))
    # Normalisasi gambar
    face_img = cv2.normalize(face_img, None, 0, 255, cv2.NORM_MINMAX)
    return face_img

def deteksi_wajah(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    wajah = facereg.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return wajah

def prediksi_umur_jenis_kelamin(face_img):
    face_img = pre_process(face_img)
    blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

    # Prediksi jenis kelamin
    gender_net.setInput(blob)
    gender_preds = gender_net.forward()
    gender = gender_list[gender_preds[0].argmax()]

    # Prediksi usia
    age_net.setInput(blob)
    age_preds = age_net.forward()
    age = age_list[age_preds[0].argmax()]

    return gender, age

def draw_label(frame, text, pos, bg_color):
    font_scale = 0.6
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, 1)
    x, y = pos
    cv2.rectangle(frame, (x, y - text_height - 10), (x + text_width, y + baseline), bg_color, cv2.FILLED)
    cv2.putText(frame, text, (x, y - 5), font, font_scale, (0, 0, 0), 1, cv2.LINE_AA)

def is_pinch(hand_landmarks):
    # Mendapatkan koordinat jari telunjuk dan ibu jari
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

    # Hitung jarak antara ibu jari dan telunjuk
    distance = math.sqrt((thumb_tip.x - index_finger_tip.x) ** 2 +
                         (thumb_tip.y - index_finger_tip.y) ** 2 +
                         (thumb_tip.z - index_finger_tip.z) ** 2)
    return distance < 0.05  # Sesuaikan threshold sesuai kebutuhan

def is_tap(hand_landmarks):
    # Mendapatkan koordinat jari telunjuk
    index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    index_finger_dip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP]

    # Hitung jarak antara ujung jari telunjuk dan DIP
    distance = math.sqrt((index_finger_tip.x - index_finger_dip.x) ** 2 +
                         (index_finger_tip.y - index_finger_dip.y) ** 2 +
                         (index_finger_tip.z - index_finger_dip.z) ** 2)
    return distance < 0.02  # Sesuaikan threshold sesuai kebutuhan

def overlay_image_alpha(img, img_overlay, pos, alpha_mask):
    x, y = pos
    y1, y2 = max(0, y), min(img.shape[0], y + img_overlay.shape[0])
    x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])

    y1o, y2o = max(0, -y), min(img_overlay.shape[0], img.shape[0] - y)
    x1o, x2o = max(0, -x), min(img_overlay.shape[1], img.shape[1] - x)

    if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
        return

    channels = img.shape[2]

    alpha = alpha_mask[y1o:y2o, x1o:x2o]
    alpha_inv = 1.0 - alpha

    for c in range(channels):
        img[y1:y2, x1:x2, c] = (alpha * img_overlay[y1o:y2o, x1o:x2o, c] +
                                alpha_inv * img[y1:y2, x1:x2, c])

def interpolate_position(start, end, alpha):
    """Interpolasi linier antara dua titik."""
    return (int(start[0] * (1 - alpha) + end[0] * alpha),
            int(start[1] * (1 - alpha) + end[1] * alpha))

def main():
    global object_center, prev_object_center, pinching, tapped
    alpha = 0.2  # Faktor interpolasi (0.0 - 1.0)
    while True:
        ret, frame = cam.read()
        if not ret:
            break

        # Deteksi wajah
        wajah = deteksi_wajah(frame)
        for (x, y, w, h) in wajah:
            face_img = frame[y:y + h, x:x + w].copy()
            gender, age = prediksi_umur_jenis_kelamin(face_img)
            
            # Gambar kotak di sekitar wajah
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Tampilkan usia dan jenis kelamin
            label = f"{gender}, {age}"
            draw_label(frame, label, (x, y), (255, 255, 255))
        
        # Deteksi tangan dan jari menggunakan MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(frame_rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
                )
                
                # Deteksi gestur pinch
                if is_pinch(hand_landmarks):
                    # Dapatkan posisi jari telunjuk
                    index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    h, w, _ = frame.shape
                    index_finger_pos = (int(index_finger_tip.x * w), int(index_finger_tip.y * h))
                    
                    # Jika objek sedang dipindahkan, perbarui posisinya dengan interpolasi
                    if pinching and object_center:
                        object_center = interpolate_position(object_center, index_finger_pos, alpha)
                    else:
                        pinching = True
                        object_center = index_finger_pos
                        prev_object_center = index_finger_pos
                        
                    draw_label(frame, "Ambil!", (50, 50), (0, 255, 0))
                else:
                    pinching = False

                # Deteksi gestur tap
                if is_tap(hand_landmarks):
                    tapped = True
                    object_center = None  # Hilangkan objek
                    draw_label(frame, "Ketuk!", (50, 50), (0, 0, 255))
                else:
                    tapped = False

        # Jika objek sedang dipindahkan, gambar objek di posisi baru
        if object_center:
            object_resized = cv2.resize(object_img, (200, 50))  # Ukuran objek tetap
            alpha_mask = object_resized[:, :, 3] / 255.0
            object_resized = object_resized[:, :, :3]
            overlay_image_alpha(frame, object_resized, object_center, alpha_mask)

        cv2.imshow("Deteksi Usia, Jenis Kelamin, dan Jari", frame)

        if cv2.waitKey(1) & 0xFF == ord('x'):
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
