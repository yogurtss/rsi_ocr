import hashlib
import json
import sqlite3
from contextlib import closing, contextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile


def json_text(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read_rows(path):
    with Path(path).open(encoding="utf-8-sig") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


@contextmanager
def atomic_output(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
    try:
        yield temporary
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_rows(path, rows):
    with atomic_output(path) as temporary, temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json_text(row) + "\n")


class Cache:
    def __init__(self, path):
        self.path = path
        with closing(sqlite3.connect(path)) as db, db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("CREATE TABLE IF NOT EXISTS calls (key TEXT PRIMARY KEY, response TEXT)")

    def get(self, key):
        with closing(sqlite3.connect(self.path)) as db:
            row = db.execute("SELECT response FROM calls WHERE key = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, key, value):
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute("INSERT OR IGNORE INTO calls VALUES (?, ?)", (key, json_text(value)))
