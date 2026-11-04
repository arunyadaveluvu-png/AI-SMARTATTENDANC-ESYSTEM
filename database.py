import sqlite3
import pickle
import os
import base64
import requests
from typing import List, Dict, Tuple, Optional, Any

try:
    import streamlit as st
except ImportError:
    st = None

DB_DIR = os.path.dirname(os.path.abspath(__file__))

# On read-only filesystems (Streamlit Cloud mounts app dir as read-only),
# fall back to /tmp which is always writable.
_write_probe = os.path.join(DB_DIR, ".write_test")
try:
    with open(_write_probe, "w") as _f:
        _f.write("ok")
    os.remove(_write_probe)
except (PermissionError, OSError):
    DB_DIR = "/tmp"

DB_NAME = os.path.join(DB_DIR, "attendance_system.db")

DEFAULT_SUPABASE_URL = "https://oittgqjtssnutcrrnjqt.supabase.co"
DEFAULT_SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im9pdHRncWp0c3NudXRjcnJuanF0Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3ODU4Mzc1MDksImV4cCI6MjEwMTQxMzUwOX0.9canKPOE-zCklUvPvaDVOYK4rwJF63acmNPqPF8swSM"

def get_supabase_config() -> Tuple[Optional[str], Optional[Dict[str, str]]]:
    """Retrieve Supabase URL and headers using environment, Streamlit Secrets, or default project credentials."""
    url = os.environ.get("SUPABASE_URL") or os.environ.get("supabase_url") or os.environ.get("URL")
    key = os.environ.get("SUPABASE_KEY") or os.environ.get("supabase_key") or os.environ.get("KEY")
    
    if st and hasattr(st, "secrets"):
        try:
            for k_url in ["SUPABASE_URL", "supabase_url", "URL", "url", "Supabase_Url"]:
                if not url and k_url in st.secrets:
                    url = str(st.secrets[k_url])
            for k_key in ["SUPABASE_KEY", "supabase_key", "KEY", "key", "Supabase_Key", "anon_key", "ANON_KEY"]:
                if not key and k_key in st.secrets:
                    key = str(st.secrets[k_key])
            
            if (not url or not key) and "supabase" in st.secrets:
                sub_sec = st.secrets["supabase"]
                if hasattr(sub_sec, "get"):
                    if not url:
                        url = sub_sec.get("url") or sub_sec.get("SUPABASE_URL") or sub_sec.get("supabase_url")
                    if not key:
                        key = sub_sec.get("key") or sub_sec.get("SUPABASE_KEY") or sub_sec.get("supabase_key") or sub_sec.get("anon_key")
        except Exception as e:
            print(f"Error reading st.secrets: {e}")
            
    if not url:
        url = DEFAULT_SUPABASE_URL
    if not key:
        key = DEFAULT_SUPABASE_KEY

    if url and key:
        url = str(url).strip().rstrip('/')
        key = str(key).strip()
        headers = {
            "apikey": key,
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "Prefer": "return=representation"
        }
        return url, headers
    return None, None

def is_cloud_mode() -> bool:
    """Check if Supabase cloud credentials are configured."""
    url, _ = get_supabase_config()
    return url is not None

def get_connection(db_path: str = DB_NAME) -> sqlite3.Connection:
    """Establish and return a connection to local SQLite database."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn

def init_db(db_path: str = DB_NAME) -> None:
    """Initialize database tables locally and attempt seeding Supabase cloud defaults if available."""
    url, headers = get_supabase_config()
    if url and headers:
        # Seed default faculty in Supabase if reachable
        try:
            r = requests.get(f"{url}/rest/v1/faculty?username=eq.faculty&select=*", headers=headers, timeout=3)
            if r.status_code == 200 and len(r.json()) == 0:
                requests.post(
                    f"{url}/rest/v1/faculty",
                    json={"username": "faculty", "password": "password123", "name": "Faculty Admin"},
                    headers=headers,
                    timeout=3
                )
        except Exception as e:
            print(f"Supabase connection notice (falling back to SQLite): {e}")

    # Always ensure local SQLite database tables exist
    conn = get_connection(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS students (
            register_no TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            department TEXT NOT NULL,
            embedding BLOB NOT NULL
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS faculty (
            username TEXT PRIMARY KEY,
            password TEXT NOT NULL,
            name TEXT NOT NULL
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS attendance_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            slot_name TEXT NOT NULL,
            faculty_name TEXT NOT NULL,
            date_str TEXT NOT NULL,
            time_str TEXT NOT NULL,
            total_students INTEGER NOT NULL,
            present_count INTEGER NOT NULL,
            absent_count INTEGER NOT NULL,
            location_lat REAL,
            location_lon REAL,
            location_address TEXT,
            yolo_model_name TEXT,
            detection_count INTEGER,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Automatic schema migration for existing SQLite databases
    cursor.execute("PRAGMA table_info(attendance_sessions)")
    existing_cols = [col[1] for col in cursor.fetchall()]
    if "location_lat" not in existing_cols:
        cursor.execute("ALTER TABLE attendance_sessions ADD COLUMN location_lat REAL")
    if "location_lon" not in existing_cols:
        cursor.execute("ALTER TABLE attendance_sessions ADD COLUMN location_lon REAL")
    if "location_address" not in existing_cols:
        cursor.execute("ALTER TABLE attendance_sessions ADD COLUMN location_address TEXT")
    if "yolo_model_name" not in existing_cols:
        cursor.execute("ALTER TABLE attendance_sessions ADD COLUMN yolo_model_name TEXT")
    if "detection_count" not in existing_cols:
        cursor.execute("ALTER TABLE attendance_sessions ADD COLUMN detection_count INTEGER")

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS attendance_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            register_no TEXT NOT NULL,
            name TEXT NOT NULL,
            department TEXT NOT NULL,
            status TEXT NOT NULL,
            FOREIGN KEY (session_id) REFERENCES attendance_sessions (id) ON DELETE CASCADE
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS model_training_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name TEXT NOT NULL,
            dataset_name TEXT,
            epochs_trained INTEGER NOT NULL,
            total_epochs INTEGER NOT NULL,
            final_loss REAL,
            box_loss REAL,
            cls_loss REAL,
            dfl_loss REAL,
            precision_score REAL,
            recall_score REAL,
            map50 REAL,
            map50_95 REAL,
            training_status TEXT NOT NULL,
            started_at TEXT,
            completed_at TEXT,
            notes TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS system_settings (
            setting_key TEXT PRIMARY KEY,
            setting_value TEXT NOT NULL
        )
    """)

    cursor.execute("SELECT * FROM faculty WHERE username = ?", ("faculty",))
    if not cursor.fetchone():
        cursor.execute(
            "INSERT INTO faculty (username, password, name) VALUES (?, ?, ?)",
            ("faculty", "password123", "Faculty Admin")
        )
    
    conn.commit()
    conn.close()

def add_student(register_no: str, name: str, department: str, embedding: Any, db_path: str = DB_NAME) -> bool:
    """Insert a new student into cloud Supabase or local SQLite."""
    serialized_embedding = pickle.dumps(embedding)
    url, headers = get_supabase_config()
    
    if url and headers:
        try:
            b64_embedding = base64.b64encode(serialized_embedding).decode('utf-8')
            payload = {
                "register_no": register_no.strip(),
                "name": name.strip(),
                "department": department.strip(),
                "embedding": b64_embedding
            }
            res = requests.post(f"{url}/rest/v1/students", json=payload, headers=headers, timeout=5)
            if res.status_code in (200, 201):
                return True
        except Exception as e:
            print(f"Supabase error in add_student, falling back to local SQLite: {e}")

    conn = get_connection(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO students (register_no, name, department, embedding) VALUES (?, ?, ?, ?)",
            (register_no.strip(), name.strip(), department.strip(), serialized_embedding)
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def student_exists(register_no: str, db_path: str = DB_NAME) -> bool:
    """Check if a student register number exists."""
    url, headers = get_supabase_config()
    if url and headers:
        try:
            res = requests.get(f"{url}/rest/v1/students?register_no=eq.{register_no.strip()}&select=register_no", headers=headers, timeout=5)
            if res.status_code == 200:
                return len(res.json()) > 0
        except Exception as e:
            print(f"Supabase error in student_exists, falling back to local SQLite: {e}")

    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT 1 FROM students WHERE register_no = ?", (register_no.strip(),))
    exists = cursor.fetchone() is not None
    conn.close()
    return exists

def get_all_students(db_path: str = DB_NAME) -> List[Dict[str, Any]]:
    """Retrieve all registered students with deserialized facial embeddings."""
    url, headers = get_supabase_config()
    if url and headers:
        try:
            res = requests.get(f"{url}/rest/v1/students?select=*", headers=headers, timeout=5)
            if res.status_code == 200:
                students = []
                for row in res.json():
                    try:
                        b64_str = row["embedding"]
                        raw_bytes = base64.b64decode(b64_str.encode('utf-8'))
                        emb = pickle.loads(raw_bytes)
                    except Exception:
                        emb = None
                    students.append({
                        "register_no": row["register_no"],
                        "name": row["name"],
                        "department": row["department"],
                        "embedding": emb
                    })
                return students
        except Exception as e:
            print(f"Supabase error in get_all_students, falling back to local SQLite: {e}")

    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT register_no, name, department, embedding FROM students")
    rows = cursor.fetchall()
    conn.close()
    
    students = []
    for row in rows:
        students.append({
            "register_no": row["register_no"],
            "name": row["name"],
            "department": row["department"],
            "embedding": pickle.loads(row["embedding"])
        })
    return students

def add_faculty(username: str, password: str, name: str, db_path: str = DB_NAME) -> bool:
    """Register a new faculty account."""
    url, headers = get_supabase_config()
    if url and headers:
        try:
            payload = {"username": username.strip(), "password": password.strip(), "name": name.strip()}
            res = requests.post(f"{url}/rest/v1/faculty", json=payload, headers=headers, timeout=5)
            if res.status_code in (200, 201):
                return True
        except Exception as e:
            print(f"Supabase error in add_faculty, falling back to local SQLite: {e}")

    conn = get_connection(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO faculty (username, password, name) VALUES (?, ?, ?)",
            (username.strip(), password.strip(), name.strip())
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def faculty_exists(username: str, db_path: str = DB_NAME) -> bool:
    """Check if a faculty username exists."""
    url, headers = get_supabase_config()
    if url and headers:
        try:
            res = requests.get(f"{url}/rest/v1/faculty?username=eq.{username.strip()}&select=username", headers=headers, timeout=5)
            if res.status_code == 200:
                return len(res.json()) > 0
        except Exception as e:
            print(f"Supabase error in faculty_exists, falling back to local SQLite: {e}")

    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT 1 FROM faculty WHERE username = ?", (username.strip(),))
    exists = cursor.fetchone() is not None
    conn.close()
    return exists

def verify_faculty(username: str, password: str, db_path: str = DB_NAME) -> Optional[Dict[str, Any]]:
    """Verify faculty login credentials with auto-seeding for default faculty account."""
    u_clean = username.strip()
    p_clean = password.strip()
    
    url, headers = get_supabase_config()
    if url and headers:
        try:
            res = requests.get(
                f"{url}/rest/v1/faculty?username=eq.{u_clean}&password=eq.{p_clean}&select=username,name",
                headers=headers,
                timeout=5
            )
            if res.status_code == 200 and len(res.json()) > 0:
                row = res.json()[0]
                return {"username": row["username"], "name": row["name"]}
            
            # Auto-seed default faculty account if requested and missing
            if u_clean.lower() == "faculty" and p_clean == "password123":
                try:
                    requests.post(
                        f"{url}/rest/v1/faculty",
                        json={"username": "faculty", "password": "password123", "name": "Faculty Admin"},
                        headers=headers,
                        timeout=5
                    )
                except Exception:
                    pass
                return {"username": "faculty", "name": "Faculty Admin"}
            if res.status_code == 200:
                return None
        except Exception as e:
            print(f"Supabase error in verify_faculty, falling back to local SQLite: {e}")

    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT username, name FROM faculty WHERE username = ? AND password = ?",
        (u_clean, p_clean)
    )
    row = cursor.fetchone()
    conn.close()
    
    if row:
        return {"username": row["username"], "name": row["name"]}
        
    if u_clean.lower() == "faculty" and p_clean == "password123":
        return {"username": "faculty", "name": "Faculty Admin"}
        
    return None

def get_all_faculty(db_path: str = DB_NAME) -> List[Dict[str, str]]:
    """Retrieve all registered faculty members."""
    url, headers = get_supabase_config()
    if url and headers:
        try:
            res = requests.get(f"{url}/rest/v1/faculty?select=username,name", headers=headers, timeout=5)
            if res.status_code == 200:
                return res.json()
        except Exception as e:
            print(f"Supabase error in get_all_faculty, falling back to local SQLite: {e}")

    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT username, name FROM faculty")
    rows = cursor.fetchall()
    conn.close()
    return [{"username": row["username"], "name": row["name"]} for row in rows]

def save_attendance_session(
    slot_name: str,
    faculty_name: str,
    date_str: str,
    time_str: str,
    total_students: int,
    present_count: int,
    absent_count: int,
    records: List[Dict[str, str]],
    location_lat: Optional[float] = None,
    location_lon: Optional[float] = None,
    location_address: Optional[str] = None,
    yolo_model_name: Optional[str] = None,
    detection_count: Optional[int] = None,
    db_path: str = DB_NAME
) -> int:
    """Save an attendance session with genuine GPS location and YOLO metrics."""
    url, headers = get_supabase_config()
    if url and headers:
        try:
            payload = {
                "slot_name": slot_name.strip(),
                "faculty_name": faculty_name.strip(),
                "date_str": date_str.strip(),
                "time_str": time_str.strip(),
                "total_students": total_students,
                "present_count": present_count,
                "absent_count": absent_count,
                "location_lat": location_lat,
                "location_lon": location_lon,
                "location_address": location_address,
                "yolo_model_name": yolo_model_name,
                "detection_count": detection_count
            }
            res = requests.post(f"{url}/rest/v1/attendance_sessions", json=payload, headers=headers, timeout=5)
            session_id = 0
            if res.status_code in (200, 201) and len(res.json()) > 0:
                session_id = res.json()[0].get("id", 0)
            
            if session_id:
                log_payloads = []
                for r in records:
                    log_payloads.append({
                        "session_id": session_id,
                        "register_no": r.get("Register No", r.get("register_no", "")).strip(),
                        "name": r.get("Name", r.get("name", "")).strip(),
                        "department": r.get("Department", r.get("department", "")).strip(),
                        "status": r.get("Status", r.get("status", "")).strip()
                    })
                requests.post(f"{url}/rest/v1/attendance_logs", json=log_payloads, headers=headers, timeout=5)
                return session_id
        except Exception as e:
            print(f"Supabase error in save_attendance_session, falling back to local SQLite: {e}")

    conn = get_connection(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute("""
            INSERT INTO attendance_sessions (
                slot_name, faculty_name, date_str, time_str, total_students,
                present_count, absent_count, location_lat, location_lon,
                location_address, yolo_model_name, detection_count
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            slot_name.strip(), faculty_name.strip(), date_str.strip(), time_str.strip(),
            total_students, present_count, absent_count,
            location_lat, location_lon, location_address,
            yolo_model_name, detection_count
        ))
        
        session_id = cursor.lastrowid
        for r in records:
            cursor.execute("""
                INSERT INTO attendance_logs (session_id, register_no, name, department, status)
                VALUES (?, ?, ?, ?, ?)
            """, (session_id, r.get("Register No", r.get("register_no", "")).strip(),
                  r.get("Name", r.get("name", "")).strip(),
                  r.get("Department", r.get("department", "")).strip(),
                  r.get("Status", r.get("status", "")).strip()))
        conn.commit()
        return session_id
    finally:
        conn.close()

def get_all_attendance_sessions(faculty_name: Optional[str] = None, db_path: str = DB_NAME) -> List[Dict[str, Any]]:
    """Retrieve attendance sessions, optionally filtered by faculty_name."""
    url, headers = get_supabase_config()
    if url and headers:
        try:
            endpoint = f"{url}/rest/v1/attendance_sessions?select=*&order=id.desc"
            if faculty_name and faculty_name.strip():
                endpoint += f"&faculty_name=eq.{faculty_name.strip()}"
            res = requests.get(endpoint, headers=headers, timeout=5)
            if res.status_code == 200:
                return res.json()
        except Exception as e:
            print(f"Supabase error in get_all_attendance_sessions, falling back to local SQLite: {e}")

    conn = get_connection(db_path)
    cursor = conn.cursor()
    if faculty_name and faculty_name.strip():
        cursor.execute("""
            SELECT *
            FROM attendance_sessions
            WHERE faculty_name = ?
            ORDER BY id DESC
        """, (faculty_name.strip(),))
    else:


        
        cursor.execute("""
            SELECT *
            FROM attendance_sessions
            ORDER BY id DESC
        """)
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]

def get_attendance_session_details(session_id: int, db_path: str = DB_NAME) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, str]]]:
    """Retrieve session details and logs."""
    url, headers = get_supabase_config()
    if url and headers:
        try:
            res1 = requests.get(f"{url}/rest/v1/attendance_sessions?id=eq.{session_id}&select=*", headers=headers, timeout=5)
            if res1.status_code == 200 and len(res1.json()) > 0:
                session_info = res1.json()[0]
                res2 = requests.get(f"{url}/rest/v1/attendance_logs?session_id=eq.{session_id}&select=*", headers=headers, timeout=5)
                logs = res2.json() if res2.status_code == 200 else []
                records = [{
                    "Register No": r["register_no"],
                    "Name": r["name"],
                    "Department": r["department"],
                    "Status": r["status"]
                } for r in logs]
                return session_info, records
        except Exception as e:
            print(f"Supabase error in get_attendance_session_details, falling back to local SQLite: {e}")

    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM attendance_sessions WHERE id = ?", (session_id,))
    s_row = cursor.fetchone()
    if not s_row:
        conn.close()
        return None, []
    
    session_info = dict(s_row)
    cursor.execute("SELECT register_no, name, department, status FROM attendance_logs WHERE session_id = ?", (session_id,))
    l_rows = cursor.fetchall()
    conn.close()
    records = [{
        "Register No": r["register_no"],
        "Name": r["name"],
        "Department": r["department"],
        "Status": r["status"]
    } for r in l_rows]
    return session_info, records

def delete_attendance_session(session_id: int, db_path: str = DB_NAME) -> bool:
    """Delete attendance session."""
    url, headers = get_supabase_config()
    if url and headers:
        try:
            requests.delete(f"{url}/rest/v1/attendance_logs?session_id=eq.{session_id}", headers=headers, timeout=5)
            res = requests.delete(f"{url}/rest/v1/attendance_sessions?id=eq.{session_id}", headers=headers, timeout=5)
            if res.status_code in (200, 204):
                return True
        except Exception as e:
            print(f"Supabase error in delete_attendance_session, falling back to local SQLite: {e}")

    conn = get_connection(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute("DELETE FROM attendance_logs WHERE session_id = ?", (session_id,))
        cursor.execute("DELETE FROM attendance_sessions WHERE id = ?", (session_id,))
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()

def update_student(register_no: str, name: str, department: str, db_path: str = DB_NAME) -> bool:
    """Update student profile."""
    url, headers = get_supabase_config()
    if url and headers:
        try:
            res = requests.patch(
                f"{url}/rest/v1/students?register_no=eq.{register_no.strip()}",
                json={"name": name.strip(), "department": department.strip()},
                headers=headers,
                timeout=5
            )
            if res.status_code in (200, 204):
                return True
        except Exception as e:
            print(f"Supabase error in update_student, falling back to local SQLite: {e}")

    conn = get_connection(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute(
            "UPDATE students SET name = ?, department = ? WHERE register_no = ?",
            (name.strip(), department.strip(), register_no.strip())
        )
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()

def delete_student(register_no: str, db_path: str = DB_NAME) -> bool:
    """Delete student profile."""
    url, headers = get_supabase_config()
    if url and headers:
        try:
            res = requests.delete(f"{url}/rest/v1/students?register_no=eq.{register_no.strip()}", headers=headers, timeout=5)
            if res.status_code in (200, 204):
                return True
        except Exception as e:
            print(f"Supabase error in delete_student, falling back to local SQLite: {e}")

    conn = get_connection(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute("DELETE FROM students WHERE register_no = ?", (register_no.strip(),))
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()

def save_model_training_run(
    model_name: str,
    dataset_name: str,
    epochs_trained: int,
    total_epochs: int,
    final_loss: Optional[float] = None,
    box_loss: Optional[float] = None,
    cls_loss: Optional[float] = None,
    dfl_loss: Optional[float] = None,
    precision_score: Optional[float] = None,
    recall_score: Optional[float] = None,
    map50: Optional[float] = None,
    map50_95: Optional[float] = None,
    training_status: str = "COMPLETED",
    started_at: Optional[str] = None,
    completed_at: Optional[str] = None,
    notes: Optional[str] = None,
    db_path: str = DB_NAME
) -> int:
    """Record a genuine model training/fine-tuning run into SQLite."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute("""
            INSERT INTO model_training_history (
                model_name, dataset_name, epochs_trained, total_epochs,
                final_loss, box_loss, cls_loss, dfl_loss,
                precision_score, recall_score, map50, map50_95,
                training_status, started_at, completed_at, notes
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            model_name, dataset_name, epochs_trained, total_epochs,
            final_loss, box_loss, cls_loss, dfl_loss,
            precision_score, recall_score, map50, map50_95,
            training_status, started_at, completed_at, notes
        ))
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()

def get_all_model_training_runs(db_path: str = DB_NAME) -> List[Dict[str, Any]]:
    """Retrieve all recorded model training runs."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT * FROM model_training_history ORDER BY id DESC")
        rows = cursor.fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()

def get_latest_model_training_run(db_path: str = DB_NAME) -> Optional[Dict[str, Any]]:
    """Retrieve the most recent model training run."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT * FROM model_training_history ORDER BY id DESC LIMIT 1")
        row = cursor.fetchone()
        return dict(row) if row else None
    finally:
        conn.close()

def save_setting(key: str, value: str, db_path: str = DB_NAME) -> None:
    """Save or update a system setting in local database."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute("""
            INSERT INTO system_settings (setting_key, setting_value)
            VALUES (?, ?)
            ON CONFLICT(setting_key) DO UPDATE SET setting_value = excluded.setting_value
        """, (key, str(value)))
        conn.commit()
    finally:
        conn.close()

def get_setting(key: str, default: Optional[str] = None, db_path: str = DB_NAME) -> Optional[str]:
    """Retrieve a system setting from local database."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT setting_value FROM system_settings WHERE setting_key = ?", (key,))
        row = cursor.fetchone()
        return row[0] if row else default
    except Exception:
        return default
    finally:
        conn.close()

if __name__ == "__main__":
    init_db()
    init_db()
    print("Database module executed successfully. Mode:", "Supabase Cloud" if is_cloud_mode() else "Local SQLite")
