#!/usr/bin/env python3
"""Generate emotion-labeled stories using Gemini API.

171 emotions x 100 topics x 10 stories = 171,000 stories.
Concurrent API calls (up to 100), SQLite WAL for storage,
saves both raw API output and parsed stories.

Run:
    python -m full_replication.generate_stories
    python -m full_replication.generate_stories --test
    python -m full_replication.generate_stories --workers 50
"""

import argparse
import os
import re
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from google import genai
from google.genai import types
from tqdm import tqdm

from full_replication.config import EMOTIONS, TOPICS, STORY_PROMPT

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "stories.db")
MODEL = "gemini-2.0-flash-lite"
STORIES_PER_CALL = 10

# Appended to Anthropic's prompt to enforce strict output format
FORMAT_SUFFIX = """

OUTPUT FORMAT: Start directly with [story 1] — no preamble, no introductions, no explanations, no commentary. Output ONLY the stories separated by [story N] markers. Nothing else."""

# Thread-local storage for DB connections
_local = threading.local()
_db_lock = threading.Lock()


def get_db():
    """Get thread-local DB connection with WAL mode."""
    if not hasattr(_local, "conn"):
        _local.conn = sqlite3.connect(DB_PATH, timeout=30)
        _local.conn.execute("PRAGMA journal_mode=WAL")
        _local.conn.execute("PRAGMA busy_timeout=10000")
    return _local.conn


def init_db():
    """Create tables if they don't exist."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS api_calls (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            emotion TEXT NOT NULL,
            topic_idx INTEGER NOT NULL,
            topic TEXT NOT NULL,
            raw_response TEXT,
            status TEXT DEFAULT 'pending',
            error TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(emotion, topic_idx)
        );
        CREATE TABLE IF NOT EXISTS stories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            api_call_id INTEGER NOT NULL,
            emotion TEXT NOT NULL,
            topic_idx INTEGER NOT NULL,
            topic TEXT NOT NULL,
            story_idx INTEGER NOT NULL,
            text TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (api_call_id) REFERENCES api_calls(id),
            UNIQUE(emotion, topic_idx, story_idx)
        );
        CREATE INDEX IF NOT EXISTS idx_stories_emotion ON stories(emotion);
        CREATE INDEX IF NOT EXISTS idx_api_calls_status ON api_calls(status);
        CREATE TABLE IF NOT EXISTS stories_clean (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            api_call_id INTEGER NOT NULL,
            emotion TEXT NOT NULL,
            topic_idx INTEGER NOT NULL,
            topic TEXT NOT NULL,
            story_idx INTEGER NOT NULL,
            text TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (api_call_id) REFERENCES api_calls(id),
            UNIQUE(emotion, topic_idx, story_idx)
        );
        CREATE INDEX IF NOT EXISTS idx_stories_clean_emotion ON stories_clean(emotion);
    """)
    conn.commit()
    conn.close()


def get_completed():
    """Return set of (emotion, topic_idx) already in stories_clean."""
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    rows = conn.execute(
        "SELECT DISTINCT emotion, topic_idx FROM stories_clean"
    ).fetchall()
    conn.close()
    return set(rows)


_PREAMBLE_RE = re.compile(
    r'^(Here\s+are|Here\s+is|Below\s+are|These\s+are|The\s+following|I\'ve\s+written|Sure|Okay)',
    re.IGNORECASE
)


def is_preamble(text):
    """Check if text is model preamble rather than an actual story."""
    return bool(_PREAMBLE_RE.match(text.strip()))


def clean_story(text):
    """Strip leading markdown bold/headers and trailing junk."""
    text = text.strip()
    # Remove leading **Title** or ## Title lines
    text = re.sub(r'^(?:\*\*[^*]+\*\*|#{1,3}\s+.+)\s*\n', '', text).strip()
    return text


def parse_stories(text, expected_count=10):
    """Parse model output into individual stories."""
    min_stories = max(2, expected_count // 2)

    # Strategy 1: [story N] markers
    parts = re.split(r'\[story\s*\d+\]', text, flags=re.IGNORECASE)
    parts = [p.strip() for p in parts if p.strip() and len(p.strip()) > 50]
    if len(parts) >= min_stories:
        return parts

    # Strategy 2: Numbered patterns
    parts = re.split(r'(?:^|\n)\s*(?:\*{0,2}(?:Story\s+)?\d+[\.\):\*]{1,3}\s*\*{0,2})', text, flags=re.IGNORECASE)
    parts = [p.strip() for p in parts if p.strip() and len(p.strip()) > 50]
    if len(parts) >= min_stories:
        return parts

    # Strategy 3: Double newline separation
    parts = re.split(r'\n\s*\n', text)
    parts = [p.strip() for p in parts if p.strip() and len(p.strip()) > 50]
    if len(parts) >= min_stories:
        return parts

    # Fallback
    if len(text.strip()) > 100:
        return [text.strip()]
    return []


def generate_one(client, emotion, topic_idx, topic):
    """Generate stories for one emotion x topic, save to DB."""
    prompt = STORY_PROMPT.format(
        n_stories=STORIES_PER_CALL,
        topic=topic,
        emotion=emotion,
    ) + FORMAT_SUFFIX

    db = get_db()
    raw_response = None
    error = None
    status = "error"

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
        stories = parse_stories(raw_response, STORIES_PER_CALL)

        if not stories:
            error = "no stories parsed"
            status = "error"
        else:
            status = "done"

    except Exception as e:
        error = str(e)[:500]
        stories = []

    # Save to DB
    with _db_lock:
        try:
            cursor = db.execute(
                """INSERT OR REPLACE INTO api_calls
                   (emotion, topic_idx, topic, raw_response, status, error)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (emotion, topic_idx, topic, raw_response, status, error),
            )
            api_call_id = cursor.lastrowid

            for i, story_text in enumerate(stories):
                db.execute(
                    """INSERT OR REPLACE INTO stories
                       (api_call_id, emotion, topic_idx, topic, story_idx, text)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (api_call_id, emotion, topic_idx, topic, i, story_text),
                )

            # Write clean versions (skip preamble, clean formatting)
            clean_idx = 0
            for story_text in stories:
                if is_preamble(story_text):
                    continue
                cleaned = clean_story(story_text)
                if len(cleaned) > 50:
                    db.execute(
                        """INSERT OR REPLACE INTO stories_clean
                           (api_call_id, emotion, topic_idx, topic, story_idx, text)
                           VALUES (?, ?, ?, ?, ?, ?)""",
                        (api_call_id, emotion, topic_idx, topic, clean_idx, cleaned),
                    )
                    clean_idx += 1

            db.commit()
        except Exception as e:
            db.rollback()
            error = str(e)[:500]

    return {
        "emotion": emotion,
        "topic_idx": topic_idx,
        "n_stories": len(stories),
        "status": status,
        "error": error,
    }


def backfill_clean():
    """Re-parse all existing stories into stories_clean table."""
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")

    # Get all api_calls that are done
    calls = conn.execute(
        "SELECT id, emotion, topic_idx, topic FROM api_calls WHERE status = 'done'"
    ).fetchall()

    cleaned_total = 0
    skipped_total = 0

    for api_call_id, emotion, topic_idx, topic in calls:
        rows = conn.execute(
            "SELECT story_idx, text FROM stories WHERE api_call_id = ? ORDER BY story_idx",
            (api_call_id,)
        ).fetchall()

        clean_idx = 0
        for _, story_text in rows:
            if is_preamble(story_text):
                skipped_total += 1
                continue
            cleaned = clean_story(story_text)
            if len(cleaned) > 50:
                conn.execute(
                    """INSERT OR REPLACE INTO stories_clean
                       (api_call_id, emotion, topic_idx, topic, story_idx, text)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (api_call_id, emotion, topic_idx, topic, clean_idx, cleaned),
                )
                clean_idx += 1
                cleaned_total += 1

    conn.commit()
    print(f"Backfill complete: {cleaned_total} clean stories, {skipped_total} preambles skipped")
    conn.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Single call test")
    parser.add_argument("--workers", type=int, default=100, help="Concurrent workers")
    parser.add_argument("--backfill", action="store_true", help="Backfill stories_clean from existing stories")
    args = parser.parse_args()

    init_db()

    if args.backfill:
        backfill_clean()
        return

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY not found in .env")
        return
    completed = get_completed()

    # Build work queue
    tasks = []
    for emotion in EMOTIONS:
        topics = TOPICS
        for ti, topic in enumerate(topics):
            if (emotion, ti) not in completed:
                tasks.append((emotion, ti, topic))

    total = len(EMOTIONS) * len(TOPICS)
    done = total - len(tasks)

    if args.test:
        tasks = tasks[:1]
        print(f"TEST MODE: 1 call only")
    print(f"=== Story Generation (Gemini API) ===")
    print(f"Total: {total} calls ({STORIES_PER_CALL} stories each)")
    print(f"Done: {done}, Remaining: {len(tasks)}")
    print(f"Workers: {min(args.workers, len(tasks))}")

    if not tasks:
        print("All stories already generated.")
        return

    client = genai.Client(api_key=api_key)

    errors = 0
    total_stories = 0
    workers = min(args.workers, len(tasks))

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(generate_one, client, emotion, ti, topic): (emotion, ti)
            for emotion, ti, topic in tasks
        }

        with tqdm(total=len(tasks), desc="Generating", unit="call") as pbar:
            for future in as_completed(futures):
                result = future.result()
                total_stories += result["n_stories"]
                if result["status"] == "error":
                    errors += 1
                pbar.update(1)
                pbar.set_postfix(
                    stories=total_stories,
                    errors=errors,
                    rate=f"{total_stories/(pbar.n or 1)*STORIES_PER_CALL:.0f}/call"
                )

    # Summary
    conn = sqlite3.connect(DB_PATH, timeout=30)
    total_stories_db = conn.execute("SELECT COUNT(*) FROM stories").fetchone()[0]
    total_clean_db = conn.execute("SELECT COUNT(*) FROM stories_clean").fetchone()[0]
    total_calls_done = conn.execute("SELECT COUNT(*) FROM api_calls WHERE status='done'").fetchone()[0]
    total_errors = conn.execute("SELECT COUNT(*) FROM api_calls WHERE status='error'").fetchone()[0]
    conn.close()

    print(f"\n=== COMPLETE ===")
    print(f"API calls: {total_calls_done} done, {total_errors} errors")
    print(f"Stories (raw): {total_stories_db}")
    print(f"Stories (clean): {total_clean_db}")
    print(f"DB: {DB_PATH}")


if __name__ == "__main__":
    main()
