import requests
import cv2
import numpy as np
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from ultralytics import YOLO
from collections import deque
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

MODEL_PATH = "face_model.xml"
DATABASE_PATH = "database"
yolo = YOLO("yolov8n.pt")
history = deque(maxlen=5)
recognizer = None
id_to_name = {}


def load_recognizer():
    global recognizer, id_to_name
    if os.path.exists(MODEL_PATH) and os.path.exists("labels.npy"):
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read(MODEL_PATH)
        label_dict = np.load("labels.npy", allow_pickle=True).item()
        id_to_name = {v: k for k, v in label_dict.items()}
        print("[AI] Recognizer loaded")
    else:
        print("[AI] No model found, run /train first")


load_recognizer()


def preprocess_face(gray):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def build_db():
    rec = cv2.face.LBPHFaceRecognizer_create()
    faces, labels = [], []
    label_dict, current_id = {}, 0

    for root, dirs, files in os.walk(DATABASE_PATH):
        folder_name = os.path.basename(root)
        if folder_name == os.path.basename(DATABASE_PATH):
            continue

        for file in files:
            if file.lower().endswith((".jpg", ".png", ".jpeg")):
                path = os.path.join(root, file)
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue

                detected = face_cascade.detectMultiScale(
                    img, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
                )

                if folder_name not in label_dict:
                    label_dict[folder_name] = current_id
                    current_id += 1

                if len(detected) == 0:
                    face = cv2.resize(img, (200, 200))
                    face = preprocess_face(face)
                    faces.append(face)
                    labels.append(label_dict[folder_name])
                else:
                    for (x, y, w, h) in detected:
                        face = img[y:y+h, x:x+w]
                        face = cv2.resize(face, (200, 200))
                        face = preprocess_face(face)
                        faces.append(face)
                        labels.append(label_dict[folder_name])

    if len(faces) == 0:
        return False, "No faces found in database"

    rec.train(faces, np.array(labels))
    rec.save(MODEL_PATH)
    np.save("labels.npy", label_dict)
    return True, label_dict


def recognize_face(person_crop, threshold=65):
    if recognizer is None:
        return None

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


class TrainRequest(BaseModel):
    person_name: str
    s3_urls: List[str]


@app.post("/train")
def train(body: TrainRequest):
    person_dir = os.path.join(DATABASE_PATH, body.person_name)
    os.makedirs(person_dir, exist_ok=True)

    downloaded = []
    for url in body.s3_urls:
        try:
            res = requests.get(url, timeout=10)
            if res.status_code == 200:
                ext = url.split(".")[-1].split("?")[0]
                filename = f"{len(downloaded)}.{ext}"
                file_path = os.path.join(person_dir, filename)
                with open(file_path, "wb") as f:
                    f.write(res.content)
                downloaded.append(file_path)
        except Exception as e:
            print(f"[AI] Failed to download {url}: {e}")

    if len(downloaded) == 0:
        return JSONResponse(
            status_code=400,
            content={"error": "No images could be downloaded from S3"}
        )

    success, result = build_db()
    if not success:
        return JSONResponse(status_code=400, content={"error": result})

    load_recognizer()
    return JSONResponse({
        "message": f"Trained {len(downloaded)} photos for {body.person_name}",
        "labels": result
    })


@app.get("/dataset")
def list_dataset():
    if not os.path.exists(DATABASE_PATH):
        return {"people": []}
    people = [
        {
            "name": d,
            "photos": len(os.listdir(os.path.join(DATABASE_PATH, d)))
        }
        for d in os.listdir(DATABASE_PATH)
        if os.path.isdir(os.path.join(DATABASE_PATH, d))
    ]
    return {"people": people}


@app.delete("/dataset/{name}")
def delete_person(name: str):
    import shutil
    person_dir = os.path.join(DATABASE_PATH, name)
    if os.path.exists(person_dir):
        shutil.rmtree(person_dir)
        return JSONResponse({"message": f"Deleted {name}"})
    return JSONResponse(status_code=404, content={"error": "Person not found"})


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": recognizer is not None}


def generate_frames():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        results = yolo(frame, conf=0.5)
        current_labels = []

        for r in results:
            for box in r.boxes:
                if int(box.cls[0]) != 0:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                h, w = frame.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                result = recognize_face(crop)
                if result is None:
                    continue

                name, conf, (fx, fy, fw, fh), matched = result
                ax1, ay1 = x1 + fx, y1 + fy
                ax2, ay2 = ax1 + fw, ay1 + fh
                score = max(0, 100 - conf)
                current_labels.append(name)

                color = (0, 255, 0) if matched else (0, 0, 255)
                cv2.rectangle(frame, (ax1, ay1), (ax2, ay2), color, 2)
                cv2.putText(frame, f"{name} ({score:.1f}%)",
                            (ax1, ay1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        if current_labels:
            history.append(current_labels[0])
        if history:
            stable_name = max(set(history), key=history.count)
            cv2.putText(frame, f"Stable: {stable_name}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        _, buffer = cv2.imencode(".jpg", frame)
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
        )

    cap.release()


@app.get("/stream")
def stream():
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )