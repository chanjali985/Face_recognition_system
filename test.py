import mysql.connector
from mysql.connector import Error

try:
    connection = mysql.connector.connect(
        host="172.17.48.1",   # ✅ Windows host IP for WSL
        user="root",          # ⚙️ Your MySQL username
        password="1234",      # ⚙️ Your MySQL password
        database="face_attendance"
    )

    if connection.is_connected():
        print("[✅] Successfully connected to MySQL Database!")
        cursor = connection.cursor()
        cursor.execute("SHOW TABLES;")
        tables = cursor.fetchall()
        print("[📋] Tables found in DB:")
        for t in tables:
            print("   -", t[0])

except Error as e:
    print(f"[❌] MySQL connection error: {e}")

finally:
    if 'connection' in locals() and connection.is_connected():
        cursor.close()
        connection.close()
        print("[🔒] Connection closed.")
