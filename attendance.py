import csv
from datetime import datetime
import os
import mysql.connector

# ✅ File for local backup
ATTENDANCE_FILE = "attendance.csv"

# ✅ MySQL connection details
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "=1234",   # ⚠️ Replace this
    "database": "face_attendance"
}

def mark_attendance(name):
    now = datetime.now()
    time_str = now.strftime('%H:%M:%S')
    date_str = now.strftime('%Y-%m-%d')

    # Save to CSV
    with open('attendance.csv', 'a') as f:
        f.write(f'\n{name},{time_str},{date_str}')

    # Save to MySQL
    try:
        connection = mysql.connector.connect(
            host='localhost',
            user='root',
            password='yourpassword',  # <-- replace with your MySQL password
            database='face_attendance'
        )
        cursor = connection.cursor()

        # Get student_id
        cursor.execute("SELECT id FROM students WHERE name = %s", (name,))
        student = cursor.fetchone()

        if student:
            student_id = student[0]
        else:
            # If student not found, insert new
            cursor.execute("INSERT INTO students (name) VALUES (%s)", (name,))
            connection.commit()
            student_id = cursor.lastrowid

        # Insert attendance
        cursor.execute(
            "INSERT INTO attendance (student_id, date, time) VALUES (%s, %s, %s)",
            (student_id, date_str, time_str)
        )
        connection.commit()
        cursor.close()
        connection.close()
        print(f"[✅] Attendance saved to DB for {name}")

    except Exception as e:
        print(f"[❌] Database error: {e}")