#!/usr/bin/env python3
"""Generate neutral dialogues for denoising baseline using Gemini API.

100 topics x 12 dialogues = 1,200 neutral dialogues.
Concurrent API calls, SQLite WAL storage.

Run:
    python -m full_replication.generate_neutral
    python -m full_replication.generate_neutral --test
    python -m full_replication.generate_neutral --workers 50
"""

import argparse
import os
import re
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from google import genai
from google.genai import types
from tqdm import tqdm

from full_replication.config import TOPICS, NEUTRAL_PROMPT, N_NEUTRAL_PER_TOPIC

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "neutral.db")
MODEL = "gemini-2.0-flash-lite"

FORMAT_SUFFIX = """

OUTPUT FORMAT: Start directly with the first dialogue — no preamble, no introductions, no explanations, no commentary. Separate dialogues with a blank line then [dialogue N]. Nothing else."""

_local = threading.local()
_db_lock = threading.Lock()


def get_db():
    if not hasattr(_local, "conn"):
        _local.conn = sqlite3.connect(DB_PATH, timeout=30)
        _local.conn.execute("PRAGMA journal_mode=WAL")
        _local.conn.execute("PRAGMA busy_timeout=10000")
    return _local.conn


def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS api_calls (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            topic_idx INTEGER NOT NULL,
            topic TEXT NOT NULL,
            raw_response TEXT,
            status TEXT DEFAULT 'pending',
            error TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(topic_idx)
        );
        CREATE TABLE IF NOT EXISTS dialogues (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            api_call_id INTEGER NOT NULL,
            topic_idx INTEGER NOT NULL,
            topic TEXT NOT NULL,
            dialogue_idx INTEGER NOT NULL,
            text TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (api_call_id) REFERENCES api_calls(id),
            UNIQUE(topic_idx, dialogue_idx)
        );
        CREATE INDEX IF NOT EXISTS idx_dialogues_topic ON dialogues(topic_idx);
    """)
    conn.commit()
    conn.close()


def get_completed():
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    rows = conn.execute(
        "SELECT DISTINCT topic_idx FROM dialogues"
    ).fetchall()
    conn.close()
    return set(r[0] for r in rows)


_PREAMBLE_RE = re.compile(
    r'^(Here\s+are|Here\s+is|Below\s+are|These\s+are|The\s+following|I\'ve\s+written|Sure|Okay)',
    re.IGNORECASE
)


def parse_dialogues(text, expected_count=12):
    """Parse model output into individual dialogues."""
    # Split on [dialogue N] markers
    parts = re.split(r'\[dialogue\s*\d+\]', text, flags=re.IGNORECASE)
    parts = [p.strip() for p in parts if p.strip() and len(p.strip()) > 30]
    if len(parts) >= expected_count // 2:
        return parts

    # Split on numbered patterns
    parts = re.split(r'(?:^|\n)\s*(?:\*{0,2}Dialogue\s+\d+\*{0,2}[:\.]?|\d+[\.\)]\s)', text, flags=re.IGNORECASE)
    parts = [p.strip() for p in parts if p.strip() and len(p.strip()) > 30]
    if len(parts) >= expected_count // 2:
        return parts

    # Split on triple newlines
    parts = re.split(r'\n\s*\n\s*\n', text)
    parts = [p.strip() for p in parts if p.strip() and len(p.strip()) > 30]
    if len(parts) >= 2:
        return parts

    if len(text.strip()) > 50:
        return [text.strip()]
    return []


def convert_speakers(text):
    """Convert Person/AI to Human/Assistant per Anthropic's method."""
    text = re.sub(r'^Person:', 'Human:', text, flags=re.MULTILINE)
    text = re.sub(r'^AI:', 'Assistant:', text, flags=re.MULTILINE)
    return text


def generate_one(client, topic_idx, topic):
    prompt = NEUTRAL_PROMPT.format(n_stories=N_NEUTRAL_PER_TOPIC, topic=topic) + FORMAT_SUFFIX

    db = get_db()
    raw_response = None
    error = None
    status = "error"
    dialogues = []

    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.9,
                top_p=0.95,
                top_k=64,
                max_output_tokens=4096,
            ),
        )
        raw_response = response.text
        parsed = parse_dialogues(raw_response, N_NEUTRAL_PER_TOPIC)

        # Filter preamble and convert speakers
        for p in parsed:
            if not _PREAMBLE_RE.match(p.strip()):
                dialogues.append(convert_speakers(p))

        if not dialogues:
            error = "no dialogues parsed"
        else:
            status = "done"

    except Exception as e:
        error = str(e)[:500]

    with _db_lock:
        try:
            cursor = db.execute(
                """INSERT OR REPLACE INTO api_calls
                   (topic_idx, topic, raw_response, status, error)
                   VALUES (?, ?, ?, ?, ?)""",
                (topic_idx, topic, raw_response, status, error),
            )
            api_call_id = cursor.lastrowid

            for i, dialogue_text in enumerate(dialogues):
                db.execute(
                    """INSERT OR REPLACE INTO dialogues
                       (api_call_id, topic_idx, topic, dialogue_idx, text)
                       VALUES (?, ?, ?, ?, ?)""",
                    (api_call_id, topic_idx, topic, i, dialogue_text),
                )
            db.commit()
        except Exception as e:
            db.rollback()
            error = str(e)[:500]

    return {
        "topic_idx": topic_idx,
        "n_dialogues": len(dialogues),
        "status": status,
        "error": error,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Single call test")
    parser.add_argument("--workers", type=int, default=50, help="Concurrent workers")
    args = parser.parse_args()

    init_db()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY not found in .env")
        return

    completed = get_completed()

    tasks = []
    for ti, topic in enumerate(TOPICS):
        if ti not in completed:
            tasks.append((ti, topic))

    total = len(TOPICS)
    done = total - len(tasks)

    if args.test:
        tasks = tasks[:1]
        print(f"TEST MODE: 1 call only")

    print(f"=== Neutral Dialogue Generation (Gemini API) ===")
    print(f"Total: {total} calls ({N_NEUTRAL_PER_TOPIC} dialogues each)")
    print(f"Done: {done}, Remaining: {len(tasks)}")
    print(f"Workers: {min(args.workers, len(tasks))}")

    if not tasks:
        print("All neutral dialogues already generated.")
        return

    client = genai.Client(api_key=api_key)

    errors = 0
    total_dialogues = 0
    workers = min(args.workers, len(tasks))

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(generate_one, client, ti, topic): ti
            for ti, topic in tasks
        }

        with tqdm(total=len(tasks), desc="Generating", unit="call") as pbar:
            for future in as_completed(futures):
                result = future.result()
                total_dialogues += result["n_dialogues"]
                if result["status"] == "error":
                    errors += 1
                pbar.update(1)
                pbar.set_postfix(
                    dialogues=total_dialogues,
                    errors=errors,
                )

    conn = sqlite3.connect(DB_PATH, timeout=30)
    total_db = conn.execute("SELECT COUNT(*) FROM dialogues").fetchone()[0]
    total_done = conn.execute("SELECT COUNT(*) FROM api_calls WHERE status='done'").fetchone()[0]
    total_errors = conn.execute("SELECT COUNT(*) FROM api_calls WHERE status='error'").fetchone()[0]
    conn.close()

    print(f"\n=== COMPLETE ===")
    print(f"API calls: {total_done} done, {total_errors} errors")
    print(f"Dialogues in DB: {total_db}")
    print(f"DB: {DB_PATH}")


if __name__ == "__main__":
    main()
