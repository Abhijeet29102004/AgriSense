import sqlite3
import uuid
from fastapi import APIRouter, Query, HTTPException, Form

router = APIRouter()

DB_PATH = "chat_history.db"

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

@router.post("/user/new_chat")
def new_chat(user_id: str = Form(...), title: str = Form("New Chat")):
    chat_id = str(uuid.uuid4())
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO chats (id, user_id, title) VALUES (?, ?, ?)",
        (chat_id, user_id, title)
    )
    conn.commit()
    conn.close()
    return {"chat_id": chat_id, "title": title}

@router.get("/user/chats")
def get_chats(user_id: str = Query(...)):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, title, created_at FROM chats WHERE user_id = ? ORDER BY created_at DESC",
        (user_id,)
    )
    chats = [{"chat_id": row["id"], "title": row["title"], "created_at": row["created_at"]} for row in cursor.fetchall()]
    conn.close()
    return {"chats": chats}

@router.get("/user/chat_history")
def get_chat_history(chat_id: str = Query(...)):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT role, content, timestamp FROM messages WHERE chat_id = ? ORDER BY id ASC",
        (chat_id,)
    )
    rows = cursor.fetchall()
    conn.close()
    history = [
        {"role": row["role"], "content": row["content"], "timestamp": row["timestamp"]}
        for row in rows
    ]
    return {"history": history}

@router.get("/user/history")
def get_user_history(user_id: str = Query(...), limit: int = Query(50)):
    """
    Get chat history for a user.
    """
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT role, content, timestamp FROM messages WHERE user_id = ? ORDER BY id DESC LIMIT ?",
        (user_id, limit)
    )
    rows = cursor.fetchall()
    conn.close()
    history = [
        {"role": row["role"], "content": row["content"], "timestamp": row["timestamp"]}
        for row in rows
    ]
    return {"history": history}

@router.delete("/user/history")
def delete_user_history(user_id: str = Query(...)):
    """
    Delete all chat history for a user.
    """
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM messages WHERE user_id = ?", (user_id,))
    conn.commit()
    conn.close()
    return {"success": True, "message": "Chat history deleted."}

@router.post("/user/update_chat_title")
def update_chat_title(chat_id: str = Form(...), title: str = Form(...)):
    # Derive a short title from the user's input
    # Use the first 8 words or up to 40 characters, whichever is shorter
    words = title.strip().split()
    short_title = " ".join(words[:8])
    if len(short_title) > 20:
        short_title = short_title[:40].rstrip() + "..."
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE chats SET title = ? WHERE id = ?",
        (short_title, chat_id)
    )
    conn.commit()
    conn.close()
    return {"success": True}