# Smart Student Attendance Marker

This script uses your laptop webcam to recognize faces and automatically log their name and time into a CSV file.

## What it does
- Detects faces with OpenCV Haar cascades and LBPH face recognition.
- Opens your webcam and matches live faces to known names.
- Writes a new attendance row to `attendance.csv` the first time a person is seen each day.

## Requirements
- Python 3.8+
- `opencv-contrib-python`
- `pandas`
- `openpyxl`

## Install
```bash
python -m pip install opencv-contrib-python pandas openpyxl
```

> On Windows, this project uses `opencv-contrib-python` so you do not need `face_recognition` or `dlib`.

## Usage
1. Create a `known_faces/` folder next to `attendance_marker_combined.py`.
2. Add one image per person, named like `Alice.jpg` or `Bob.png`.
3. Run:
```bash
python attendance_marker_combined.py
```

Optional Excel output:
```bash
python attendance_marker_combined.py --excel
```

Custom output path:
```bash
python attendance_marker_combined.py --output attendance.xlsx
```

4. The webcam window will appear. Press `q` to stop.
5. Attendance is automatically appended to the selected output file.

## Notes
- The script only marks attendance once per person per day.
- If a known face is not detected, add a clearer photo with a single front-facing face.
- If no faces are found in your images, the script prints a warning and skips them.
