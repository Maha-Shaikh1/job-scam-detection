import sqlite3

conn = sqlite3.connect("database.db")
cursor = conn.cursor()

cursor.execute(
    "UPDATE users SET role='admin' WHERE username='maya'"
)

conn.commit()
conn.close()

print("User is now ADMIN ✅")
