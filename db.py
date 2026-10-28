"""SQLite storage: users, analysis results, alerts, settings."""
import hashlib, hmac, os, sqlite3, json
from contextlib import contextmanager
from .config import DB_PATH, DEFAULT_SETTINGS, DEFAULT_ADMIN


@contextmanager
def conn():
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    try:
        yield c
        c.commit()
    finally:
        c.close()


def hash_pw(password, salt=None):
    salt = salt or os.urandom(16).hex()
    h = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), 120_000).hex()
    return f"{salt}${h}"


def check_pw(password, stored):
    salt, _ = stored.split("$")
    return hmac.compare_digest(hash_pw(password, salt), stored)


def init_db():
    with conn() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS users(
            id INTEGER PRIMARY KEY, username TEXT UNIQUE, email TEXT,
            pw TEXT, role TEXT DEFAULT 'user', active INTEGER DEFAULT 1,
            created TEXT DEFAULT CURRENT_TIMESTAMP);
        CREATE TABLE IF NOT EXISTS results(
            id INTEGER PRIMARY KEY, user_id INTEGER, aoi_name TEXT, bbox TEXT,
            date_before TEXT, date_after TEXT, source TEXT,
            changed_area_ha REAL, risk_score REAL, risk_level TEXT,
            impact TEXT, image_path TEXT, created TEXT DEFAULT CURRENT_TIMESTAMP);
        CREATE TABLE IF NOT EXISTS alerts(
            id INTEGER PRIMARY KEY, result_id INTEGER, message TEXT, level TEXT,
            status TEXT DEFAULT 'Pending', field_notes TEXT DEFAULT '',
            created TEXT DEFAULT CURRENT_TIMESTAMP);
        CREATE TABLE IF NOT EXISTS settings(key TEXT PRIMARY KEY, value TEXT);
        """)
        for k, v in DEFAULT_SETTINGS.items():
            c.execute("INSERT OR IGNORE INTO settings VALUES(?,?)", (k, str(v)))
        if not c.execute("SELECT 1 FROM users WHERE username=?", (DEFAULT_ADMIN[0],)).fetchone():
            c.execute("INSERT INTO users(username,email,pw,role) VALUES(?,?,?,'admin')",
                      (DEFAULT_ADMIN[0], "admin@riverguard.local", hash_pw(DEFAULT_ADMIN[1])))


# ---- accounts -------------------------------------------------------------
def create_user(username, email, password):
    if not username.strip():
        return False, "Username is required."
    if len(password) < 6:
        return False, "Password must be at least 6 characters."
    try:
        with conn() as c:
            c.execute("INSERT INTO users(username,email,pw) VALUES(?,?,?)",
                      (username.strip(), email.strip(), hash_pw(password)))
        return True, "Account created. Please log in."
    except sqlite3.IntegrityError:
        return False, "Username already exists."


def login(username, password):
    with conn() as c:
        r = c.execute("SELECT * FROM users WHERE username=?", (username.strip(),)).fetchone()
    if r and r["active"] and check_pw(password, r["pw"]):
        return dict(r)
    return None


def list_users():
    with conn() as c:
        return [dict(r) for r in c.execute(
            "SELECT id,username,email,role,active,created FROM users ORDER BY id")]


def update_user(uid, **fields):
    with conn() as c:
        for k, v in fields.items():
            if k in ("role", "active"):
                c.execute(f"UPDATE users SET {k}=? WHERE id=?", (v, uid))


def delete_user(uid):
    with conn() as c:
        c.execute("DELETE FROM users WHERE id=? AND username!=?", (uid, DEFAULT_ADMIN[0]))


# ---- settings -------------------------------------------------------------
def get_settings():
    with conn() as c:
        return {r["key"]: float(r["value"]) for r in c.execute("SELECT * FROM settings")}


def set_setting(key, value):
    with conn() as c:
        c.execute("REPLACE INTO settings VALUES(?,?)", (key, str(value)))


# ---- results & alerts -----------------------------------------------------
def save_result(user_id, aoi_name, bbox, d1, d2, source, area, risk, level, impact, image_path):
    with conn() as c:
        cur = c.execute(
            "INSERT INTO results(user_id,aoi_name,bbox,date_before,date_after,source,"
            "changed_area_ha,risk_score,risk_level,impact,image_path) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            (user_id, aoi_name, json.dumps(bbox), d1, d2, source, area, risk, level,
             json.dumps(impact), image_path))
        return cur.lastrowid


def create_alert(result_id, message, level):
    with conn() as c:
        cur = c.execute("INSERT INTO alerts(result_id,message,level) VALUES(?,?,?)",
                        (result_id, message, level))
        return cur.lastrowid


def list_alerts():
    with conn() as c:
        return [dict(r) for r in c.execute(
            "SELECT a.*, r.aoi_name, r.changed_area_ha, r.risk_score, r.image_path "
            "FROM alerts a JOIN results r ON r.id=a.result_id ORDER BY a.id DESC")]


def update_alert(aid, status, notes):
    with conn() as c:
        c.execute("UPDATE alerts SET status=?, field_notes=? WHERE id=?", (status, notes, aid))


def list_results(user_id=None):
    q = ("SELECT r.*, u.username FROM results r LEFT JOIN users u ON u.id=r.user_id "
         + ("WHERE r.user_id=? " if user_id else "") + "ORDER BY r.id DESC")
    with conn() as c:
        return [dict(r) for r in c.execute(q, (user_id,) if user_id else ())]
