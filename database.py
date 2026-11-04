"""
database.py — JSON file-based storage for AI Smart Attendance System.

All data is stored as plain JSON files inside a `.local/` folder:
  .local/students.json
  .local/faculty.json
  .local/attendance_sessions.json
  .local/attendance_logs.json
  .local/model_training_history.json
  .local/settings.json

On Streamlit Cloud the app directory is read-only, so .local/ is placed
inside /tmp which is always writable.
"""

import os
import json
import pickle
import base64
import datetime
from typing import List, Dict, Tuple, Optional, Any

# ---------------------------------------------------------------------------
# Storage directory — writable on both local and Streamlit Cloud
# ---------------------------------------------------------------------------

_APP_DIR = os.path.dirname(os.path.abspath(__file__))

# Probe whether the app dir itself is writable (not on Streamlit Cloud)
_probe = os.path.join(_APP_DIR, ".write_test")
try:
    with open(_probe, "w") as _f:
        _f.write("ok")
    os.remove(_probe)
    _BASE = _APP_DIR
except (PermissionError, OSError):
    _BASE = "/tmp"

LOCAL_DIR = os.path.join(_BASE, ".local")
os.makedirs(LOCAL_DIR, exist_ok=True)

# File paths
_STUDENTS_FILE   = os.path.join(LOCAL_DIR, "students.json")
_FACULTY_FILE    = os.path.join(LOCAL_DIR, "faculty.json")
_SESSIONS_FILE   = os.path.join(LOCAL_DIR, "attendance_sessions.json")
_LOGS_FILE       = os.path.join(LOCAL_DIR, "attendance_logs.json")
_TRAINING_FILE   = os.path.join(LOCAL_DIR, "model_training_history.json")
_SETTINGS_FILE   = os.path.join(LOCAL_DIR, "settings.json")

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _read(path: str) -> Any:
    """Read a JSON file; return [] or {} if missing/empty."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return [] if not path.endswith("settings.json") else {}


def _write(path: str, data: Any) -> None:
    """Atomically write data to a JSON file."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def _next_id(records: List[Dict]) -> int:
    """Return max existing id + 1 (or 1 if empty)."""
    if not records:
        return 1
    return max(r.get("id", 0) for r in records) + 1

# ---------------------------------------------------------------------------
# Public API — same signatures as the old SQLite/Supabase version
# ---------------------------------------------------------------------------

def init_db(**_kwargs) -> None:
    """Ensure storage directory exists and seed default faculty account."""
    os.makedirs(LOCAL_DIR, exist_ok=True)
    faculty = _read(_FACULTY_FILE)
    if not any(f["username"] == "faculty" for f in faculty):
        faculty.append({"username": "faculty", "password": "password123", "name": "Faculty Admin"})
        _write(_FACULTY_FILE, faculty)


def is_cloud_mode() -> bool:
    """Always False — we use local JSON storage."""
    return False


# ── Students ────────────────────────────────────────────────────────────────

def add_student(register_no: str, name: str, department: str, embedding: Any, **_kwargs) -> bool:
    """Register a new student with their face embedding."""
    students = _read(_STUDENTS_FILE)
    if any(s["register_no"] == register_no.strip() for s in students):
        return False  # duplicate
    b64_emb = base64.b64encode(pickle.dumps(embedding)).decode("utf-8")
    students.append({
        "register_no": register_no.strip(),
        "name": name.strip(),
        "department": department.strip(),
        "embedding": b64_emb
    })
    _write(_STUDENTS_FILE, students)
    return True


def student_exists(register_no: str, **_kwargs) -> bool:
    """Check whether a student register number already exists."""
    students = _read(_STUDENTS_FILE)
    return any(s["register_no"] == register_no.strip() for s in students)


def get_all_students(**_kwargs) -> List[Dict[str, Any]]:
    """Return all students with deserialized face embeddings."""
    students = _read(_STUDENTS_FILE)
    result = []
    for s in students:
        try:
            emb = pickle.loads(base64.b64decode(s["embedding"].encode("utf-8")))
        except Exception:
            emb = None
        result.append({
            "register_no": s["register_no"],
            "name": s["name"],
            "department": s["department"],
            "embedding": emb
        })
    return result


def update_student(register_no: str, name: str, department: str, **_kwargs) -> bool:
    """Update an existing student's name and department."""
    students = _read(_STUDENTS_FILE)
    for s in students:
        if s["register_no"] == register_no.strip():
            s["name"] = name.strip()
            s["department"] = department.strip()
            _write(_STUDENTS_FILE, students)
            return True
    return False


def delete_student(register_no: str, **_kwargs) -> bool:
    """Delete a student record."""
    students = _read(_STUDENTS_FILE)
    new = [s for s in students if s["register_no"] != register_no.strip()]
    if len(new) == len(students):
        return False
    _write(_STUDENTS_FILE, new)
    return True


# ── Faculty ─────────────────────────────────────────────────────────────────

def add_faculty(username: str, password: str, name: str, **_kwargs) -> bool:
    """Register a new faculty account."""
    faculty = _read(_FACULTY_FILE)
    if any(f["username"] == username.strip() for f in faculty):
        return False
    faculty.append({"username": username.strip(), "password": password.strip(), "name": name.strip()})
    _write(_FACULTY_FILE, faculty)
    return True


def faculty_exists(username: str, **_kwargs) -> bool:
    """Check whether a faculty username exists."""
    faculty = _read(_FACULTY_FILE)
    return any(f["username"] == username.strip() for f in faculty)


def verify_faculty(username: str, password: str, **_kwargs) -> Optional[Dict[str, Any]]:
    """Verify faculty credentials; auto-seed default account if needed."""
    u, p = username.strip(), password.strip()
    faculty = _read(_FACULTY_FILE)
    for f in faculty:
        if f["username"] == u and f["password"] == p:
            return {"username": f["username"], "name": f["name"]}
    # Auto-seed default faculty on first login attempt
    if u.lower() == "faculty" and p == "password123":
        init_db()
        return {"username": "faculty", "name": "Faculty Admin"}
    return None


def get_all_faculty(**_kwargs) -> List[Dict[str, str]]:
    """Return all faculty accounts (username + name, no passwords)."""
    faculty = _read(_FACULTY_FILE)
    return [{"username": f["username"], "name": f["name"]} for f in faculty]


# ── Attendance Sessions ──────────────────────────────────────────────────────

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
    **_kwargs
) -> int:
    """Save an attendance session and its per-student logs."""
    sessions = _read(_SESSIONS_FILE)
    logs = _read(_LOGS_FILE)

    session_id = _next_id(sessions)
    session = {
        "id": session_id,
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
        "detection_count": detection_count,
        "created_at": datetime.datetime.now().isoformat()
    }
    sessions.append(session)
    _write(_SESSIONS_FILE, sessions)

    log_id = _next_id(logs)
    for r in records:
        logs.append({
            "id": log_id,
            "session_id": session_id,
            "register_no": r.get("Register No", r.get("register_no", "")).strip(),
            "name": r.get("Name", r.get("name", "")).strip(),
            "department": r.get("Department", r.get("department", "")).strip(),
            "status": r.get("Status", r.get("status", "")).strip()
        })
        log_id += 1
    _write(_LOGS_FILE, logs)
    return session_id


def get_all_attendance_sessions(faculty_name: Optional[str] = None, **_kwargs) -> List[Dict[str, Any]]:
    """Return all sessions (newest first), optionally filtered by faculty."""
    sessions = _read(_SESSIONS_FILE)
    if faculty_name and faculty_name.strip():
        sessions = [s for s in sessions if s.get("faculty_name") == faculty_name.strip()]
    return list(reversed(sessions))


def get_attendance_session_details(
    session_id: int, **_kwargs
) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, str]]]:
    """Return (session_info, records_list) for a given session id."""
    sessions = _read(_SESSIONS_FILE)
    session_info = next((s for s in sessions if s["id"] == session_id), None)
    if not session_info:
        return None, []

    logs = _read(_LOGS_FILE)
    records = [
        {
            "Register No": lg["register_no"],
            "Name": lg["name"],
            "Department": lg["department"],
            "Status": lg["status"]
        }
        for lg in logs if lg.get("session_id") == session_id
    ]
    return session_info, records


def delete_attendance_session(session_id: int, **_kwargs) -> bool:
    """Delete a session and all its logs."""
    sessions = _read(_SESSIONS_FILE)
    new_sessions = [s for s in sessions if s["id"] != session_id]
    if len(new_sessions) == len(sessions):
        return False
    _write(_SESSIONS_FILE, new_sessions)

    logs = _read(_LOGS_FILE)
    _write(_LOGS_FILE, [lg for lg in logs if lg.get("session_id") != session_id])
    return True


# ── Model Training History ───────────────────────────────────────────────────

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
    **_kwargs
) -> int:
    """Record a model training run."""
    runs = _read(_TRAINING_FILE)
    run_id = _next_id(runs)
    runs.append({
        "id": run_id,
        "model_name": model_name,
        "dataset_name": dataset_name,
        "epochs_trained": epochs_trained,
        "total_epochs": total_epochs,
        "final_loss": final_loss,
        "box_loss": box_loss,
        "cls_loss": cls_loss,
        "dfl_loss": dfl_loss,
        "precision_score": precision_score,
        "recall_score": recall_score,
        "map50": map50,
        "map50_95": map50_95,
        "training_status": training_status,
        "started_at": started_at,
        "completed_at": completed_at,
        "notes": notes
    })
    _write(_TRAINING_FILE, runs)
    return run_id


def get_all_model_training_runs(**_kwargs) -> List[Dict[str, Any]]:
    """Return all training runs newest first."""
    runs = _read(_TRAINING_FILE)
    return list(reversed(runs))


def get_latest_model_training_run(**_kwargs) -> Optional[Dict[str, Any]]:
    """Return the most recent training run."""
    runs = _read(_TRAINING_FILE)
    return runs[-1] if runs else None


# ── System Settings ──────────────────────────────────────────────────────────

def save_setting(key: str, value: str, **_kwargs) -> None:
    """Persist a key-value setting."""
    settings = _read(_SETTINGS_FILE)
    settings[key] = str(value)
    _write(_SETTINGS_FILE, settings)


def get_setting(key: str, default: Optional[str] = None, **_kwargs) -> Optional[str]:
    """Read a setting value, returning default if not found."""
    settings = _read(_SETTINGS_FILE)
    return settings.get(key, default)
