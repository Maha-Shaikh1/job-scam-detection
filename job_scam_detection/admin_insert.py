import sqlite3

conn = sqlite3.connect("jobshield.db")
cursor = conn.cursor()

cursor.execute("""
INSERT INTO users (username, password, role)
VALUES (?, ?, ?)
""", ("admin", "admin123", "admin"))

conn.commit()
conn.close()

print("Admin inserted")
