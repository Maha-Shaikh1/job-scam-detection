def init_db():
    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()

    # USERS TABLE
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        password TEXT,
        role TEXT DEFAULT 'user',
        failed_attempts INTEGER DEFAULT 0,
        account_locked INTEGER DEFAULT 0
    )
    """)

    # ACTIVITY LOGS
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS activity_logs(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT,
        activity TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # SCAM HISTORY
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS scam_history(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT,
        job_title TEXT,
        result TEXT,
        confidence_score REAL,
        date DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)

    conn.commit()
    conn.close()

init_db()
