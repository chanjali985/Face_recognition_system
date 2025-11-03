import mysql.connector

def get_connection():
    """Connect to MySQL database."""
    connection = mysql.connector.connect(
        host="localhost",       # change if remote DB
        user="root",            # your MySQL username
        password="1234",  # your MySQL password
        database="face_attendance" # DB name
    )
    return connection
