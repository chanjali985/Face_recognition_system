import os
import cv2
import pickle
import numpy as np
import time
import face_recognition
import mysql.connector
from mysql.connector import Error
from datetime import datetime
from attendance import mark_attendance
from eval_metrics import evaluate_face_recognition


# -------------------------------
# ✅ MySQL Database Setup
# -------------------------------
def mark_attendance_in_db(name):
    try:
        connection = mysql.connector.connect(
            host="localhost",
            user="root",          # your MySQL username
            password="yourpassword",  # your MySQL password
            database="face_attendance"
        )
        cursor = connection.cursor()

        print(f"[DB] ✅ Connected to MySQL")

        # 1️⃣ Check if student exists
        cursor.execute("SELECT id FROM students WHERE name = %s", (name,))
        result = cursor.fetchone()

        if result:
            student_id = result[0]
            now = datetime.now()
            date = now.date()
            time = now.time()

            cursor.execute("""
                INSERT INTO attendance (student_id, date, time)
                VALUES (%s, %s, %s)
            """, (student_id, date, time))

            connection.commit()
            print(f"[DB] 📥 Attendance recorded for {name} at {time}")
        else:
            print(f"[DB] ⚠️ Student '{name}' not found in DB — please register in 'students' table.")

    except mysql.connector.Error as err:
        print(f"[DB] ❌ Error: {err}")
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'connection' in locals() and connection.is_connected():
            connection.close()


# -------------------------------
# Helper Functions
# -------------------------------
def load_images_from_folder(folder):
    """Load all valid images from a given folder."""
    valid_exts = ('.jpg', '.jpeg', '.png')
    images = []

    if not os.path.exists(folder):
        print(f"[❌] Folder not found: {folder}")
        return images

    for file in os.listdir(folder):
        if file.lower().endswith(valid_exts):
            img_path = os.path.join(folder, file)
            img = cv2.imread(img_path)
            if img is not None:
                images.append(img)
            else:
                print(f"[⚠️] Could not read image: {file}")

    if not images:
        print(f"[⚠️] No valid images found in folder: {folder}")

    return images


def load_encodings(encode_path="Encodefile.p"):
    """Load face encodings and student IDs."""
    if not os.path.exists(encode_path):
        raise FileNotFoundError(f"{encode_path} not found! Run EncodeGenerator.py first.")

    print("[ℹ️] Loading face encodings...")
    with open(encode_path, 'rb') as file:
        encodeListKnown, studentIds = pickle.load(file)
    print(f"[✅] Encodings loaded for {len(studentIds)} users: {studentIds}")
    return encodeListKnown, studentIds


def load_background(path):
    """Load background image."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Background image not found: {path}")
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not load background from {path}")
    return img


# -------------------------------
# Main Function
# -------------------------------
def main():
    # Setup webcam
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    # Load assets
    imgBackground = load_background('Resources/background.PNG')
    imgModeList = load_images_from_folder('Resources/modes')
    encodeListKnown, studentIds = load_encodings()

    if len(imgModeList) == 0:
        raise ValueError("[❌] No mode images found in Resources/modes!")

    # Placement regions
    x1, y1, x2, y2 = 41, 112, 529, 486
    X1, Y1, X2, Y2 = 616, 40, 911, 502
    webcam_width, webcam_height = x2 - x1, y2 - y1
    node_image_width, node_image_height = X2 - X1, Y2 - Y1

    prev_time = 0
    print("[🚀] Starting Face Recognition System...")

    while True:
        success, img = cap.read()
        if not success:
            print("[⚠️] Webcam read failed.")
            continue

        # Reduce frame size for faster processing
        imgS = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        faceCurFrame = face_recognition.face_locations(imgS)
        encodeCurrFrame = face_recognition.face_encodings(imgS, faceCurFrame)

        imgBackgroundCopy = imgBackground.copy()
        imgBackgroundCopy[y1:y2, x1:x2] = cv2.resize(img, (webcam_width, webcam_height))
        imgBackgroundCopy[Y1:Y2, X1:X2] = cv2.resize(imgModeList[0], (node_image_width, node_image_height))

        for encodeFace, faceLoc in zip(encodeCurrFrame, faceCurFrame):
            y1_, x2_, y2_, x1_ = [v * 4 for v in faceLoc]
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)

            name, color = "Unknown", (0, 0, 255)

            if matches and len(matches) > 0:
                matchIndex = np.argmin(faceDis)
                if matches[matchIndex]:
                    name = studentIds[matchIndex]
                    color = (0, 255, 0)
                    y_true = [studentIds[matchIndex]]
                    y_pred = [studentIds[matchIndex]]

                    print(f"[😀] Recognized face: {name}")

                    # ✅ Mark attendance in CSV
                    mark_attendance(name)

                    # ✅ Save attendance in DB
                    save_attendance_to_db(name)

                else:
                    y_true = [studentIds[matchIndex]]
                    y_pred = ["Unknown"]

                # ✅ Evaluate recognition metrics
                evaluate_face_recognition(y_true, y_pred)

            cv2.rectangle(imgBackgroundCopy, (x1_, y1_), (x2_, y2_), color, 2)
            cv2.putText(imgBackgroundCopy, name, (x1_, y1_ - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # FPS display
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time else 0
        prev_time = curr_time
        cv2.putText(imgBackgroundCopy, f"FPS: {int(fps)}", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        # Show window
        cv2.imshow("Face Recognition System", imgBackgroundCopy)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n[👋] Exiting system...")
            break

    cap.release()
    cv2.destroyAllWindows()


# -------------------------------
# Entry Point
# -------------------------------
if __name__ == "__main__":
    main()
