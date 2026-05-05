import cv2
import os
import argparse
import numpy as np
from ultralytics import YOLO
from collections import deque

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

MODEL_PATH = "face_model.xml"


# 🔥 Improve lighting robustness
def preprocess_face(gray):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def build_db(database_path="database", model_path=MODEL_PATH):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    faces, labels = [], []
    label_dict, current_id = {}, 0

    for root, dirs, files in os.walk(database_path):
        folder_name = os.path.basename(root)
        if folder_name == os.path.basename(database_path):
            continue

        print(f"[DB] Folder: {folder_name}")
        for file in files:
            if file.lower().endswith((".jpg", ".png", ".jpeg")):
                path = os.path.join(root, file)
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

                detected = face_cascade.detectMultiScale(
                    img, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
                )

                if len(detected) == 0:
                    face = cv2.resize(img, (200, 200))
                    face = preprocess_face(face)
                    faces.append(face)
                else:
                    for (x, y, w, h) in detected:
                        face = img[y:y+h, x:x+w]
                        face = cv2.resize(face, (200, 200))
                        face = preprocess_face(face)
                        faces.append(face)

                if folder_name not in label_dict:
                    label_dict[folder_name] = current_id
                    current_id += 1
                labels.extend([label_dict[folder_name]] * len(detected or [0]))

    recognizer.train(faces, np.array(labels))
    recognizer.save(model_path)

    print(f"[DB] Saved → {model_path}")
    np.save("labels.npy", label_dict)


def recognize_face(person_crop, recognizer, id_to_name, threshold=65):
    gray = cv2.cvtColor(person_crop, cv2.COLOR_BGR2GRAY)

    detected = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
    )

    if len(detected) == 0:
        return None

    best_result = None

    for (x, y, w, h) in detected:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (200, 200))
        face = preprocess_face(face)

        id_, conf = recognizer.predict(face)
        name = id_to_name.get(id_, "Unknown")

        if best_result is None or conf < best_result[1]:
            best_result = (name, conf, (x, y, w, h))

    name, conf, (x, y, w, h) = best_result

    matched = conf < threshold
    if not matched:
        name = "Unknown"

    return name, conf, (x, y, w, h), matched


def main(source, build_db_flag=False):
    if build_db_flag:
        build_db()
        return

    if not os.path.exists(MODEL_PATH):
        print("Run with --build-db first")
        return

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(MODEL_PATH)

    label_dict = np.load("labels.npy", allow_pickle=True).item()
    id_to_name = {v: k for k, v in label_dict.items()}

    print("[SYSTEM] Ready")

    model = YOLO("yolov8n.pt")

    cap = cv2.VideoCapture(source)
    cap.set(3, 480)
    cap.set(4, 360)

    history = deque(maxlen=5)  # 🔥 smoothing

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        results = model(frame, conf=0.5)

        current_labels = []

        for r in results:
            for box in r.boxes:
                if int(box.cls[0]) != 0:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # 🔥 safe crop
                h, w = frame.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                result = recognize_face(crop, recognizer, id_to_name)

                if result is None:
                    continue

                name, conf, (fx, fy, fw, fh), matched = result

                # absolute face box
                ax1 = x1 + fx
                ay1 = y1 + fy
                ax2 = ax1 + fw
                ay2 = ay1 + fh

                score = max(0, 100 - conf)

                current_labels.append(name)

                color = (0, 255, 0) if matched else (0, 0, 255)
                label = f"{name} ({score:.1f}%)"

                cv2.rectangle(frame, (ax1, ay1), (ax2, ay2), color, 2)
                cv2.putText(frame, label, (ax1, ay1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # 🔥 smoothing (major stability boost)
        if current_labels:
            history.append(current_labels[0])

        if history:
            stable_name = max(set(history), key=history.count)
            cv2.putText(frame, f"Stable: {stable_name}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        cv2.imshow("Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-db", action="store_true")
    parser.add_argument("--source", type=str, default="0")

    args = parser.parse_args()
    source = 0 if args.source == "0" else args.source

    main(source, args.build_db)