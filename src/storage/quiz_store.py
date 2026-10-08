import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

from dotenv import load_dotenv

from src.storage.dedupe import normalize

load_dotenv()

DB_PATH = os.getenv("QUIZ_DB_PATH", "quiz.db")


@contextmanager
def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def _init_db() -> None:
    with _connect() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS quizzes (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                topic      TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS questions (
                id       INTEGER PRIMARY KEY AUTOINCREMENT,
                quiz_id  INTEGER NOT NULL REFERENCES quizzes(id) ON DELETE CASCADE,
                position INTEGER NOT NULL,
                question TEXT NOT NULL,
                answer   TEXT NOT NULL,
                question_key TEXT
            );
            """
        )
        # migrate databases created before question_key existed
        cols = [r["name"] for r in conn.execute("PRAGMA table_info(questions)")]
        if "question_key" not in cols:
            conn.execute("ALTER TABLE questions ADD COLUMN question_key TEXT")
            for r in conn.execute("SELECT id, question FROM questions").fetchall():
                conn.execute(
                    "UPDATE questions SET question_key = ? WHERE id = ?",
                    (normalize(r["question"]), r["id"]),
                )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_questions_key ON questions(question_key)"
        )


_init_db()


def save_quiz(topic: str, qa_pairs: list[dict[str, str]]) -> dict:
    """Store a quiz, skipping questions that already exist in the database.

    Returns {"quiz_id": int | None, "saved": int, "skipped": int}.
    quiz_id is None when every question was a duplicate (nothing is created).
    """
    keyed = [(normalize(qa["question"]), qa) for qa in qa_pairs]

    with _connect() as conn:
        existing = {
            r["question_key"]
            for r in conn.execute(
                f"SELECT question_key FROM questions WHERE question_key IN ({','.join('?' * len(keyed))})",
                [k for k, _ in keyed],
            )
        } if keyed else set()

        new_pairs = [(k, qa) for k, qa in keyed if k not in existing]
        skipped = len(keyed) - len(new_pairs)
        if not new_pairs:
            return {"quiz_id": None, "saved": 0, "skipped": skipped}

        created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        cur = conn.execute(
            "INSERT INTO quizzes (topic, created_at) VALUES (?, ?)",
            (topic, created_at),
        )
        quiz_id = cur.lastrowid
        conn.executemany(
            "INSERT INTO questions (quiz_id, position, question, answer, question_key) VALUES (?, ?, ?, ?, ?)",
            [
                (quiz_id, i, qa["question"], qa["answer"], key)
                for i, (key, qa) in enumerate(new_pairs, start=1)
            ],
        )
    return {"quiz_id": quiz_id, "saved": len(new_pairs), "skipped": skipped}


def list_quizzes() -> list[dict]:
    """Return all quizzes (newest first) with their question counts."""
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT q.id, q.topic, q.created_at, COUNT(qs.id) AS question_count
            FROM quizzes q
            LEFT JOIN questions qs ON qs.quiz_id = q.id
            GROUP BY q.id
            ORDER BY q.id DESC
            """
        ).fetchall()
    return [dict(r) for r in rows]


def get_quiz(quiz_id: int) -> dict | None:
    """Return a quiz with all its questions, or None if it doesn't exist."""
    with _connect() as conn:
        quiz = conn.execute(
            "SELECT id, topic, created_at FROM quizzes WHERE id = ?", (quiz_id,)
        ).fetchone()
        if quiz is None:
            return None
        questions = conn.execute(
            "SELECT position, question, answer FROM questions WHERE quiz_id = ? ORDER BY position",
            (quiz_id,),
        ).fetchall()
    return {**dict(quiz), "questions": [dict(q) for q in questions]}