import cv2
import os
import argparse
import numpy as np
from ultralytics import YOLO


face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

MODEL_PATH = "face_model.xml"


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
                print(f"[DB] Processing: {path}")

                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                detected = face_cascade.detectMultiScale(img, 1.3, 5)

                for (x, y, w, h) in detected:
                    face = img[y:y+h, x:x+w]
                    face = cv2.resize(face, (200, 200))
                    faces.append(face)

                    if folder_name not in label_dict:
                        label_dict[folder_name] = current_id
                        current_id += 1
                    labels.append(label_dict[folder_name])

    recognizer.train(faces, np.array(labels))
    recognizer.save(model_path)

    print(f"[DB] Trained model saved to {model_path}")
    print(f"[DB] Labels: {label_dict}")
    np.save("labels.npy", label_dict)


def recognize_face(frame, recognizer, label_dict, threshold=70):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detected_faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(detected_faces) == 0:
        return None, None, None, None

    confidences = {}
    boxes = []

    for (x, y, w, h) in detected_faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (200, 200))
        id_, conf = recognizer.predict(face)
        name = [k for k, v in label_dict.items() if v == id_][0]
        confidences.setdefault(name, []).append(conf)
        boxes.append((x, y, w, h))

    avg_conf = {n: np.mean(c) for n, c in confidences.items()}
    best_name = min(avg_conf, key=avg_conf.get)
    best_conf = avg_conf[best_name]

    if best_conf < threshold:
        return best_name, best_conf, boxes, True
    else:
        return "Unknown", best_conf, boxes, False


def main(source, build_db_flag=False):
    if build_db_flag:
        build_db()
        return

    if not os.path.exists(MODEL_PATH):
        print("No trained model found. Run with --build-db first.")
        return

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(MODEL_PATH)
    label_dict = np.load("labels.npy", allow_pickle=True).item()
    print(f"[SYSTEM] Loaded model with labels: {label_dict}")

    model = YOLO("yolov8n.pt")
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open video source: {source}")
        return

    print("[SYSTEM] Starting video. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to grab frame.")
            break

        frame = cv2.flip(frame, 1)
        results = model(frame, stream=True)

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                if cls != 0:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                person_crop = frame[y1:y2, x1:x2]

                if person_crop.size > 0:
                    name, conf, faces, matched = recognize_face(
                        person_crop, recognizer, label_dict)

                    if faces:
                        for (fx, fy, fw, fh) in faces:

                            abs_x1 = x1 + fx
                            abs_y1 = y1 + fy
                            abs_x2 = abs_x1 + fw
                            abs_y2 = abs_y1 + fh

                            color = (0, 255, 0) if matched else (0, 0, 255)
                            label = f"{name} ({conf:.2f})"

                            cv2.rectangle(frame, (abs_x1, abs_y1),
                                          (abs_x2, abs_y2), color, 2)
                            cv2.putText(frame, label, (abs_x1, abs_y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-db", action="store_true",
                        help="Train LBPH model")
    parser.add_argument("--source", type=str, default="0",
                        help="0 for webcam or video URL")
    args = parser.parse_args()

    source = 0 if args.source == "0" else args.source
    main(source=source, build_db_flag=args.build_db)
