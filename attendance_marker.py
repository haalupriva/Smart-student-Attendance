import argparse
import glob
import os
import sys
from datetime import date, datetime

try:
    import cv2  # type: ignore[reportMissingImports]
    import face_recognition  # type: ignore[reportMissingImports]
    import numpy as np  # type: ignore[reportMissingImports]
    import pandas as pd  # type: ignore[reportMissingImports]
except ModuleNotFoundError as exc:
    missing = exc.name
    print(f"Error: missing Python module '{missing}'.")
    print("Install dependencies with:")
    print("    python -m pip install opencv-python face_recognition pandas openpyxl")
    sys.exit(1)

KNOWN_FACES_DIR = "known_faces"
DEFAULT_ATTENDANCE_FILE = "attendance.csv"


def load_known_faces(folder: str):
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
        print(f"Created '{folder}/' folder. Drop labeled face photos into it and rerun the script.")
        return [], []

    known_names = []
    known_encodings = []

    image_paths = sorted(glob.glob(os.path.join(folder, "*.*")))
    if not image_paths:
        print(f"No images found in '{folder}/'. Add face photos and rerun.")
        return known_names, known_encodings

    for image_path in image_paths:
        extension = os.path.splitext(image_path)[1].lower()
        if extension not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue

        name = os.path.splitext(os.path.basename(image_path))[0]
        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)

        if not encodings:
            print(f"Warning: no face found in '{image_path}'. Skipping.")
            continue

        known_names.append(name)
        known_encodings.append(encodings[0])
        print(f"Loaded face for '{name}' from '{image_path}'.")

    return known_names, known_encodings


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
    parser = argparse.ArgumentParser(description="Smart student attendance marker using webcam face recognition.")
    parser.add_argument("--excel", action="store_true", help="Save attendance to attendance.xlsx instead of CSV.")
    parser.add_argument("--output", type=str, default=None, help="Custom attendance file path. Extension .csv or .xlsx is supported.")
    args = parser.parse_args()

    attendance_path = args.output or ("attendance.xlsx" if args.excel else DEFAULT_ATTENDANCE_FILE)

    known_names, known_encodings = load_known_faces(KNOWN_FACES_DIR)
    if not known_encodings:
        print("No known faces available. Add face images to the folder and rerun.")
        sys.exit(1)

    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        print("Unable to open webcam. Check your camera and try again.")
        sys.exit(1)

    process_this_frame = True
    print(f"Starting webcam attendance capture. Press 'q' to quit. Writing attendance to {attendance_path}")

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Failed to capture frame from webcam. Exiting.")
            break

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = []
        face_names = []

        if process_this_frame:
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.5)
                name = "Unknown"

                face_distances = face_recognition.face_distance(known_encodings, face_encoding)
                if len(face_distances) > 0:
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = known_names[best_match_index]
                        mark_attendance(name, attendance_path)

                face_names.append(name)

        process_this_frame = not process_this_frame

        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

        cv2.imshow("Smart Attendance", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
