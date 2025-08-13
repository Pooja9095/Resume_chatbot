import sqlite3

def init_db():
    conn = sqlite3.connect('me/db.sqlite')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS qa (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT UNIQUE,
            answer TEXT
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE,
            name TEXT,
            notes TEXT
        )
    ''')
    try:
        c.execute("ALTER TABLE users ADD COLUMN questions_asked INTEGER DEFAULT 0")
    except sqlite3.OperationalError:
        # Column already exists, do nothing
        pass

    c.execute('''
        CREATE TABLE IF NOT EXISTS unknown_questions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def add_qa(question, answer):
    conn = sqlite3.connect('me/db.sqlite')
    c = conn.cursor()
    try:
        c.execute("INSERT INTO qa (question, answer) VALUES (?, ?)", (question, answer))
    except sqlite3.IntegrityError:
        # If question already exists, update the answer
        c.execute("UPDATE qa SET answer = ? WHERE question = ?", (answer, question))
    conn.commit()
    conn.close()


def get_answer(question):
    conn = sqlite3.connect('me/db.sqlite')
    c = conn.cursor()
    c.execute("SELECT answer FROM qa WHERE question = ?", (question,))
    row = c.fetchone()
    conn.close()
    return row[0] if row else None

def add_user(email, name=None, notes=None):
    conn = sqlite3.connect('me/db.sqlite')
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users (email, name, notes, questions_asked) VALUES (?, ?, ?, ?)", (email, name, notes, 0))
        conn.commit()
    except sqlite3.IntegrityError:
        pass
    conn.close()

def add_unknown_question(question):
    conn = sqlite3.connect('me/db.sqlite')
    c = conn.cursor()
    c.execute("INSERT INTO unknown_questions (question) VALUES (?)", (question,))
    conn.commit()
    conn.close()

def get_user(email):
    conn = sqlite3.connect('me/db.sqlite')
    c = conn.cursor()
    c.execute("SELECT id, questions_asked FROM users WHERE email = ?", (email,))
    row = c.fetchone()
    conn.close()
    return row  # returns (id, questions_asked) or None

def increment_questions(email):
    conn = sqlite3.connect('me/db.sqlite')
    c = conn.cursor()
    c.execute("UPDATE users SET questions_asked = questions_asked + 1 WHERE email = ?", (email,))
    conn.commit()
    conn.close()


init_db()
