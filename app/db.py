"""SQLite data layer (replaces YAML/JSONL file storage).

- users: user table (passwords remain pbkdf2 hashes, compatible with the old users.yaml)
- history: processing history table (compatible with the old processing_history.json)
- On first startup, automatically migrates from config/users.yaml and admin_data/processing_history.json
"""
import json
import os
import sqlite3
import threading
import time
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ADMIN_DATA_DIR = os.path.join(BASE_DIR, "admin_data")
DB_PATH = os.path.join(ADMIN_DATA_DIR, "fixoldimg.db")
ARCHIVE_INPUT_DIR = os.path.join(ADMIN_DATA_DIR, "archive_inputs")
ARCHIVE_OUTPUT_DIR = os.path.join(ADMIN_DATA_DIR, "archive_outputs")
LEGACY_USERS_YAML = os.path.join(BASE_DIR, "config", "users.yaml")
LEGACY_HISTORY_FILE = os.path.join(ADMIN_DATA_DIR, "processing_history.json")

HISTORY_MAX = int(os.environ.get("FIXIMG_HISTORY_MAX", "2000"))
ARCHIVE_TTL_SECONDS = int(os.environ.get("FIXIMG_ARCHIVE_TTL", str(7 * 24 * 3600)))

_write_lock = threading.Lock()
_conn = None

_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    username   TEXT PRIMARY KEY,
    password   TEXT NOT NULL,
    role       TEXT NOT NULL DEFAULT 'user',
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS history (
    id          TEXT PRIMARY KEY,
    timestamp   TEXT NOT NULL,
    user        TEXT NOT NULL,
    type        TEXT NOT NULL,
    input_path  TEXT NOT NULL,
    output_path TEXT NOT NULL,
    psnr        TEXT,
    ssim        TEXT,
    mae         TEXT
);
CREATE INDEX IF NOT EXISTS idx_history_timestamp ON history(timestamp);
"""


def get_conn():
    global _conn
    if _conn is None:
        os.makedirs(ADMIN_DATA_DIR, exist_ok=True)
        _conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=5.0)
        _conn.row_factory = sqlite3.Row
        _conn.execute("PRAGMA journal_mode=WAL")
        _conn.execute("PRAGMA busy_timeout=5000")
        _conn.execute("PRAGMA synchronous=NORMAL")
    return _conn


def init_db():
    """Create tables and run one-time migrations."""
    with _write_lock:
        conn = get_conn()
        conn.executescript(_SCHEMA)
        conn.commit()
        _migrate_users(conn)
        _migrate_history(conn)


def get_user(username):
    row = get_conn().execute(
        "SELECT * FROM users WHERE username=?", (username,)
    ).fetchone()
    return dict(row) if row else None


def list_users():
    rows = get_conn().execute(
        "SELECT username, role, created_at FROM users ORDER BY username"
    ).fetchall()
    return [dict(r) for r in rows]


def add_user(username, password_hash, role="user"):
    """Create a user; returns False if the username already exists (avoids a race on concurrent registration)."""
    with _write_lock:
        conn = get_conn()
        try:
            conn.execute(
                "INSERT INTO users(username, password, role, created_at) VALUES (?,?,?,?)",
                (username, password_hash, role, datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
            )
            conn.commit()
        except sqlite3.IntegrityError:
            conn.rollback()
            return False
    return True


def update_user(username, password_hash=None, role=None):
    with _write_lock:
        conn = get_conn()
        if password_hash is not None:
            conn.execute(
                "UPDATE users SET password=? WHERE username=?", (password_hash, username)
            )
        if role is not None:
            conn.execute("UPDATE users SET role=? WHERE username=?", (role, username))
        conn.commit()


def delete_user(username):
    with _write_lock:
        conn = get_conn()
        conn.execute("DELETE FROM users WHERE username=?", (username,))
        conn.commit()


def append_history(record):
    with _write_lock:
        conn = get_conn()
        conn.execute(
            "INSERT INTO history(id, timestamp, user, type, input_path, output_path, psnr, ssim, mae) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (
                record.get("id"),
                record.get("timestamp", ""),
                record.get("user", ""),
                record.get("type", ""),
                record.get("input_path", ""),
                record.get("output_path", ""),
                str(record.get("psnr", "")),
                str(record.get("ssim", "")),
                str(record.get("mae", "")),
            ),
        )
        conn.commit()
    _enforce_history_cap()


def _enforce_history_cap():
    if HISTORY_MAX <= 0:
        return
    with _write_lock:
        conn = get_conn()
        conn.execute(
            "DELETE FROM history WHERE id NOT IN "
            "(SELECT id FROM history ORDER BY timestamp DESC LIMIT ?)",
            (HISTORY_MAX,),
        )
        conn.commit()


def list_history(limit=HISTORY_MAX):
    rows = get_conn().execute(
        "SELECT * FROM history ORDER BY timestamp DESC, id DESC LIMIT ?", (limit,)
    ).fetchall()
    return [dict(r) for r in rows]


def get_history_record(history_id):
    row = get_conn().execute(
        "SELECT * FROM history WHERE id=?", (history_id,)
    ).fetchone()
    return dict(row) if row else None


def clear_history():
    with _write_lock:
        conn = get_conn()
        conn.execute("DELETE FROM history")
        conn.commit()


def history_stats():
    rows = list_history()
    total = len(rows)
    users = len({r["user"] for r in rows})
    type_counts = {}
    psnr_vals, ssim_vals = [], []
    for r in rows:
        tp = r.get("type") or "unknown"
        type_counts[tp] = type_counts.get(tp, 0) + 1
        try:
            psnr_raw = r.get("psnr")
            if psnr_raw not in (None, "", "∞", "N/A"):
                psnr_vals.append(float(psnr_raw))
        except (TypeError, ValueError):
            pass
        try:
            ssim_raw = r.get("ssim")
            if ssim_raw not in (None, "", "N/A"):
                ssim_vals.append(float(ssim_raw))
        except (TypeError, ValueError):
            pass
    lines = [f"Total tasks: {total}", f"Users: {users}"]
    for tp, cnt in sorted(type_counts.items(), key=lambda x: -x[1]):
        lines.append(f"  · {tp}: {cnt} times")
    if psnr_vals:
        lines.append(f"Average PSNR: {sum(psnr_vals) / len(psnr_vals):.2f}")
    if ssim_vals:
        lines.append(f"Average SSIM: {sum(ssim_vals) / len(ssim_vals):.4f}")
    return "\n".join(lines)


def purge_stale_archives(ttl_seconds=ARCHIVE_TTL_SECONDS):
    now = time.time()
    for d in (ARCHIVE_INPUT_DIR, ARCHIVE_OUTPUT_DIR):
        if not os.path.isdir(d):
            continue
        for name in os.listdir(d):
            p = os.path.join(d, name)
            try:
                if now - os.path.getmtime(p) > ttl_seconds:
                    os.remove(p)
            except OSError:
                pass


def _migrate_users(conn):
    if conn.execute("SELECT COUNT(*) FROM users").fetchone()[0] > 0:
        return
    if not os.path.exists(LEGACY_USERS_YAML):
        return
    try:
        import yaml

        with open(LEGACY_USERS_YAML, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        users = data.get("users") or {}
        created = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for username, info in users.items():
            conn.execute(
                "INSERT OR IGNORE INTO users(username,password,role,created_at) VALUES (?,?,?,?)",
                (username, info.get("password", ""), info.get("role", "user"), created),
            )
        conn.commit()
        print(f"[Migration] imported {len(users)} users to SQLite")
    except Exception as exc:  # noqa: BLE001
        print("[Migration] user import failed (skipped):", exc)


def _parse_legacy_history(content):
    """Handle legacy array / JSONL / mixed formats, deduplicating by id."""
    records = []
    text = content.strip()
    try:
        obj, end = json.JSONDecoder().raw_decode(text)
        if isinstance(obj, list):
            records.extend(r for r in obj if isinstance(r, dict))
            tail = text[end:].strip()
        else:
            tail = text
    except json.JSONDecodeError:
        tail = text
    for line in tail.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
            if isinstance(rec, dict):
                records.append(rec)
        except json.JSONDecodeError:
            continue
    seen = {}
    for r in records:
        seen[r.get("id") or f"__no_id_{len(seen)}__"] = r
    return list(seen.values())


def _migrate_history(conn):
    if conn.execute("SELECT COUNT(*) FROM history").fetchone()[0] > 0:
        return
    if not os.path.exists(LEGACY_HISTORY_FILE):
        return
    try:
        with open(LEGACY_HISTORY_FILE, "r", encoding="utf-8") as f:
            content = f.read().strip()
        if not content:
            return
        records = _parse_legacy_history(content)
        for r in records:
            conn.execute(
                "INSERT OR IGNORE INTO history(id,timestamp,user,type,input_path,output_path,psnr,ssim,mae) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    r.get("id"),
                    r.get("timestamp", ""),
                    r.get("user", ""),
                    r.get("type", ""),
                    r.get("input_path", ""),
                    r.get("output_path", ""),
                    str(r.get("psnr", "")),
                    str(r.get("ssim", "")),
                    str(r.get("mae", "")),
                ),
            )
        conn.commit()
        print(f"[Migration] imported {len(records)} history records to SQLite")
    except Exception as exc:  # noqa: BLE001
        print("[Migration] history import failed (skipped):", exc)

