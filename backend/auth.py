import sqlite3
from fastapi import APIRouter, Form, HTTPException
from passlib.hash import bcrypt
from datetime import datetime
from jose import jwt
import os

router = APIRouter()

DB_PATH = "chat_history.db"

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def create_users_table():
    conn = get_db()
    cursor = conn.cursor()
    # Create table if not exists
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        gmail TEXT UNIQUE NOT NULL,
        name TEXT NOT NULL,
        password_hash TEXT,
        oauth_provider TEXT,
        oauth_id TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)
    # --- Add missing columns if they do not exist ---
    cursor.execute("PRAGMA table_info(users);")
    columns = [col[1] for col in cursor.fetchall()]
    if "password_hash" not in columns:
        cursor.execute("ALTER TABLE users ADD COLUMN password_hash TEXT;")
    if "oauth_provider" not in columns:
        cursor.execute("ALTER TABLE users ADD COLUMN oauth_provider TEXT;")
    if "oauth_id" not in columns:
        cursor.execute("ALTER TABLE users ADD COLUMN oauth_id TEXT;")
    if "created_at" not in columns:
        cursor.execute("ALTER TABLE users ADD COLUMN created_at DATETIME DEFAULT CURRENT_TIMESTAMP;")
    conn.commit()
    conn.close()

create_users_table()

# Secret key for JWT (keep this safe in production)
JWT_SECRET = os.getenv("JWT_SECRET", "your_secret_key")
JWT_ALGORITHM = "HS256"

def create_access_token(data: dict, expires_delta: int = 3600):
    to_encode = data.copy()
    to_encode["exp"] = datetime.utcnow().timestamp() + expires_delta
    return jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALGORITHM)

# --- Signup Endpoint ---
@router.post("/signup")
def signup(
    gmail: str = Form(...),
    name: str = Form(...),
    password: str = Form(...)
):
    conn = get_db()
    cursor = conn.cursor()
    password_hash = bcrypt.hash(password)
    try:
        cursor.execute(
            "INSERT INTO users (gmail, name, password_hash) VALUES (?, ?, ?)",
            (gmail, name, password_hash)
        )
        conn.commit()
    except sqlite3.IntegrityError:
        conn.close()
        raise HTTPException(status_code=400, detail="Gmail already registered")
    conn.close()
    return {"success": True, "message": "User registered successfully"}

# --- Login Endpoint ---
@router.post("/login")
def login(
    gmail: str = Form(...),
    password: str = Form(...)
):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, password_hash FROM users WHERE gmail = ?", (gmail,))
    row = cursor.fetchone()
    conn.close()
    if row and row["password_hash"] and bcrypt.verify(password, row["password_hash"]):
        token = create_access_token({"user_id": row["id"], "gmail": gmail, "name": row["name"]})
        return {"success": True, "message": "Login successful", "token": token}
    else:
        raise HTTPException(status_code=401, detail="Invalid credentials")

# --- OAuth Endpoint ---
@router.post("/oauth")
async def oauth(
    gmail: str = Form(...),
    name: str = Form(...),
    oauth_provider: str = Form(...),
    oauth_id: str = Form(...)
):
    try:
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("SELECT id, name FROM users WHERE gmail = ?", (gmail,))
        user = cursor.fetchone()

        if user:
            token = create_access_token({"user_id": user["id"], "gmail": gmail, "name": user["name"]})
            conn.close()
            return {"success": True, "message": "OAuth login successful", "token": token}
        else:
            try:
                cursor.execute(
                    "INSERT INTO users (gmail, name, oauth_provider, oauth_id) VALUES (?, ?, ?, ?)",
                    (gmail, name, oauth_provider, oauth_id)
                )
                conn.commit()
                user_id = cursor.lastrowid
                print(f"Inserted new user with id: {user_id}")
            except sqlite3.IntegrityError as e:
                conn.close()
                print(f"IntegrityError: {e}")
                raise HTTPException(status_code=400, detail="Gmail already registered")
            conn.close()
            token = create_access_token({"user_id": user_id, "gmail": gmail, "name": name})
            return {"success": True, "message": "User registered successfully via OAuth", "token": token}
    except Exception as e:
        print(f"Exception in /oauth: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")