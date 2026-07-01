"""
Smart Student Attendance Marker (OpenCV only)

This version uses OpenCV face detection and the LBPH face recognizer.
It does not require `face_recognition`/`dlib`, so it is easier to install on Windows.

Requirements:
- Python 3.8+
- opencv-contrib-python
- pandas
- openpyxl (optional, only for .xlsx output)

Install:
    python -m pip install --upgrade pip setuptools wheel
    python -m pip install opencv-contrib-python pandas openpyxl

Usage:
1. Create a `known_faces/` folder next to this script.
2. Add one or more images per person, named like `Alice.jpg` or `Bob.png`.
3. Run:
    python attendance_marker_opencv.py

Optional Excel output:
    python attendance_marker_opencv.py --excel

Custom output path:
    python attendance_marker_opencv.py --output attendance.xlsx

Press `q` to quit the webcam window.
"""

import argparse
import glob
import os
import sys
from datetime import date, datetime

try:
    import cv2
    import numpy as np
    import pandas as pd
except ModuleNotFoundError as exc:
    missing = exc.name
    print(f"Error: missing Python module '{missing}'.")
    print("Install dependencies with:")
    print("    python -m pip install opencv-contrib-python pandas openpyxl")
    sys.exit(1)

KNOWN_FACES_DIR = "known_faces"
DEFAULT_ATTENDANCE_FILE = "attendance.csv"
FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"


def load_known_faces(folder: str):
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
        print(f"Created '{folder}/' folder. Drop labeled face photos into it and rerun the script.")
        return [], [], {}

    face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
    known_faces = []
    labels = []
    name_to_id = {}
    next_id = 0

    image_paths = sorted(glob.glob(os.path.join(folder, "*.*")))
    if not image_paths:
        print(f"No images found in '{folder}/'. Add face photos and rerun.")
        return known_faces, labels, {}

    for image_path in image_paths:
        extension = os.path.splitext(image_path)[1].lower()
        if extension not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue

        name = os.path.splitext(os.path.basename(image_path))[0]
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: could not load '{image_path}'. Skipping.")
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))

        if len(faces) == 0:
            print(f"Warning: no face found in '{image_path}'. Skipping.")
            continue

        x, y, w, h = faces[0]
        face_crop = gray[y : y + h, x : x + w]
        face_resized = cv2.resize(face_crop, (200, 200))

        if name not in name_to_id:
            name_to_id[name] = next_id
            next_id += 1

        known_faces.append(face_resized)
        labels.append(name_to_id[name])
        print(f"Loaded face for '{name}' from '{image_path}'.")

    return known_faces, labels, {v: k for k, v in name_to_id.items()}


def ensure_attendance_file(path: str):
    if not os.path.exists(path):
        df = pd.DataFrame(columns=["Name", "Date", "Time"])
        write_attendance_file(df, path)
        print(f"Created attendance file: {path}")


def read_attendance_file(path: str):
    if not os.path.exists(path):
        return pd.DataFrame(columns=["Name", "Date", "Time"])

    extension = os.path.splitext(path)[1].lower()
    if extension == ".xlsx":
        return pd.read_excel(path, engine="openpyxl")

    return pd.read_csv(path)


def write_attendance_file(df: pd.DataFrame, path: str):
    extension = os.path.splitext(path)[1].lower()
    if extension == ".xlsx":
        df.to_excel(path, index=False, engine="openpyxl")
    else:
        df.to_csv(path, index=False)


def mark_attendance(name: str, path: str):
    ensure_attendance_file(path)
    timestamp = datetime.now()
    today = date.today().isoformat()
    time_str = timestamp.strftime("%H:%M:%S")

    df = read_attendance_file(path)
    already_marked = ((df["Name"] == name) & (df["Date"] == today)).any()

    if already_marked:
        return False

    new_row = {"Name": name, "Date": today, "Time": time_str}
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    write_attendance_file(df, path)
    print(f"Attendance marked for {name} at {time_str}.")
    return True


def main():
    parser = argparse.ArgumentParser(description="Smart student attendance marker using OpenCV LBPH.")
    parser.add_argument("--excel", action="store_true", help="Save attendance to attendance.xlsx instead of CSV.")
    parser.add_argument("--output", type=str, default=None, help="Custom attendance file path. Extension .csv or .xlsx is supported.")
    args = parser.parse_args()

    attendance_path = args.output or ("attendance.xlsx" if args.excel else DEFAULT_ATTENDANCE_FILE)

    known_faces, labels, id_to_name = load_known_faces(KNOWN_FACES_DIR)
    if not known_faces:
        print("No known faces available. Add face images to the folder and rerun.")
        sys.exit(1)

    if not hasattr(cv2, "face") or not hasattr(cv2.face, "LBPHFaceRecognizer_create"):
        print("Error: OpenCV face module unavailable. Install opencv-contrib-python.")
        sys.exit(1)

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(known_faces, np.array(labels))
    print(f"Trained recognizer with {len(known_faces)} known faces: {list(id_to_name.values())}")

    face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        print("Unable to open webcam. Check your camera and try again.")
        sys.exit(1)

    print(f"Starting webcam attendance capture. Press 'q' to quit. Writing attendance to {attendance_path}")
    process_this_frame = True

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Failed to capture frame from webcam. Exiting.")
            break

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

        face_locations = []
        face_names = []

        if process_this_frame:
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
            for (x, y, w, h) in faces:
                face_crop = gray[y : y + h, x : x + w]
                face_resized = cv2.resize(face_crop, (200, 200))
                label, confidence = recognizer.predict(face_resized)
                name = id_to_name.get(label, "Unknown")
                print(f"Detected {name} with confidence {confidence:.1f}")

                if confidence < 120:
                    mark_attendance(name, attendance_path)
                else:
                    name = "Unknown"

                face_locations.append((x, y, w, h))
                face_names.append(name)

        process_this_frame = not process_this_frame

        for (x, y, w, h), name in zip(face_locations, face_names):
            x_full, y_full, w_full, h_full = x * 4, y * 4, w * 4, h * 4
            cv2.rectangle(frame, (x_full, y_full), (x_full + w_full, y_full + h_full), (0, 255, 0), 2)
            cv2.rectangle(frame, (x_full, y_full + h_full - 35), (x_full + w_full, y_full + h_full), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, name, (x_full + 6, y_full + h_full - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

        cv2.imshow("Smart Attendance", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
