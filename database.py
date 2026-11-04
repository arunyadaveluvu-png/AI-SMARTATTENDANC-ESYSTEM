"""
database.py — Plain JSON file storage for AI Smart Attendance System.
All records are saved directly to JSON files in the working directory:
  - students.json
  - faculty.json
  - attendance_sessions.json
  - attendance_logs.json
  - model_training_history.json
  - settings.json

If deployed on a read-only filesystem (like Streamlit Cloud /mount/src),
it automatically falls back to /tmp to ensure writes never fail.
"""

import os
import json
import pickle
import base64
import datetime
from typing import List, Dict, Tuple, Optional, Any

# Determine base writable directory
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
_probe = os.path.join(_APP_DIR, ".write_test")
try:
    with open(_probe, "w") as _f:
        _f.write("ok")
    os.remove(_probe)
    DATA_DIR = _APP_DIR
except (PermissionError, OSError):
    DATA_DIR = "/tmp"

# Direct JSON files in root directory (no hidden folders)
_STUDENTS_FILE = os.path.join(DATA_DIR, "students.json")
_FACULTY_FILE = os.path.join(DATA_DIR, "faculty.json")
_SESSIONS_FILE = os.path.join(DATA_DIR, "attendance_sessions.json")
_LOGS_FILE = os.path.join(DATA_DIR, "attendance_logs.json")
_TRAINING_FILE = os.path.join(DATA_DIR, "model_training_history.json")
_SETTINGS_FILE = os.path.join(DATA_DIR, "settings.json")


def _read_json(path: str) -> Any:
    """Read a JSON file safely, returning empty list/dict on error."""
    if not os.path.exists(path):
        return {} if path.endswith("settings.json") else []
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {} if path.endswith("settings.json") else []


def _write_json(path: str, data: Any) -> None:
    """Write data atomically to a JSON file."""
    tmp = path + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp, path)
    except Exception as e:
        # Fallback to direct write
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as inner_e:
            print(f"Error writing to {path}: {inner_e}")


def _next_id(items: List[Dict]) -> int:
    if not items:
        return 1
    return max(r.get("id", 0) for r in items) + 1


def init_db(**_kwargs) -> None:
    """Ensure database seed data exists (default faculty account)."""
    faculty = _read_json(_FACULTY_FILE)
    if not any(f.get("username") == "faculty" for f in faculty):
        faculty.append({
            "username": "faculty",
            "password": "password123",
            "name": "Faculty Admin"
        })
        _write_json(_FACULTY_FILE, faculty)


def is_cloud_mode() -> bool:
    return False


# ── Students ────────────────────────────────────────────────────────────────

def add_student(register_no: str, name: str, department: str, embedding: Any, **_kwargs) -> bool:
    """Register a new student with serialized facial embedding."""
    students = _read_json(_STUDENTS_FILE)
    reg_clean = register_no.strip()
    if any(s.get("register_no", "").strip().lower() == reg_clean.lower() for s in students):
        return False
    
    b64_emb = ""
    if embedding is not None:
        try:
            b64_emb = base64.b64encode(pickle.dumps(embedding)).decode("utf-8")
        except Exception:
            b64_emb = ""

    students.append({
        "register_no": reg_clean,
        "name": name.strip(),
        "department": department.strip(),
        "embedding": b64_emb
    })
    _write_json(_STUDENTS_FILE, students)
    return True


def student_exists(register_no: str, **_kwargs) -> bool:
    """Check if student register number already exists."""
    students = _read_json(_STUDENTS_FILE)
    reg_clean = register_no.strip().lower()
    return any(s.get("register_no", "").strip().lower() == reg_clean for s in students)


def get_all_students(**_kwargs) -> List[Dict[str, Any]]:
    """Return all registered students with decoded facial embeddings."""
    students = _read_json(_STUDENTS_FILE)
    result = []
    for s in students:
        emb = None
        b64_str = s.get("embedding", "")
        if b64_str:
            try:
                emb = pickle.loads(base64.b64decode(b64_str.encode("utf-8")))
            except Exception:
                emb = None
        result.append({
            "register_no": s.get("register_no", ""),
            "name": s.get("name", ""),
            "department": s.get("department", ""),
            "embedding": emb
        })
    return result


def update_student(register_no: str, name: str, department: str, **_kwargs) -> bool:
    """Update student name and department."""
    students = _read_json(_STUDENTS_FILE)
    reg_clean = register_no.strip().lower()
    for s in students:
        if s.get("register_no", "").strip().lower() == reg_clean:
            s["name"] = name.strip()
            s["department"] = department.strip()
            _write_json(_STUDENTS_FILE, students)
            return True
    return False


def delete_student(register_no: str, **_kwargs) -> bool:
    """Delete student by register number."""
    students = _read_json(_STUDENTS_FILE)
    reg_clean = register_no.strip().lower()
    new_students = [s for s in students if s.get("register_no", "").strip().lower() != reg_clean]
    if len(new_students) == len(students):
        return False
    _write_json(_STUDENTS_FILE, new_students)
    return True


# ── Faculty ─────────────────────────────────────────────────────────────────

def add_faculty(username: str, password: str, name: str, **_kwargs) -> bool:
    faculty = _read_json(_FACULTY_FILE)
    u_clean = username.strip().lower()
    if any(f.get("username", "").strip().lower() == u_clean for f in faculty):
        return False
    faculty.append({
        "username": username.strip(),
        "password": password.strip(),
        "name": name.strip()
    })
    _write_json(_FACULTY_FILE, faculty)
    return True


def faculty_exists(username: str, **_kwargs) -> bool:
    faculty = _read_json(_FACULTY_FILE)
    u_clean = username.strip().lower()
    return any(f.get("username", "").strip().lower() == u_clean for f in faculty)


def verify_faculty(username: str, password: str, **_kwargs) -> Optional[Dict[str, Any]]:
    u = username.strip().lower()
    p = password.strip()
    faculty = _read_json(_FACULTY_FILE)
    for f in faculty:
        if f.get("username", "").strip().lower() == u and f.get("password") == p:
            return {"username": f.get("username"), "name": f.get("name")}
    # Default admin credentials fallback
    if u == "faculty" and p == "password123":
        init_db()
        return {"username": "faculty", "name": "Faculty Admin"}
    return None


def get_all_faculty(**_kwargs) -> List[Dict[str, str]]:
    faculty = _read_json(_FACULTY_FILE)
    return [{"username": f.get("username", ""), "name": f.get("name", "")} for f in faculty]


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
    sessions = _read_json(_SESSIONS_FILE)
    logs = _read_json(_LOGS_FILE)

    session_id = _next_id(sessions)
    session = {
        "id": session_id,
        "slot_name": slot_name.strip(),
        "faculty_name": faculty_name.strip(),
        "date_str": date_str.strip(),
        "time_str": time_str.strip(),
        "total_students": int(total_students),
        "present_count": int(present_count),
        "absent_count": int(absent_count),
        "location_lat": location_lat,
        "location_lon": location_lon,
        "location_address": location_address,
        "yolo_model_name": yolo_model_name,
        "detection_count": detection_count,
        "created_at": datetime.datetime.now().isoformat()
    }
    sessions.append(session)
    _write_json(_SESSIONS_FILE, sessions)

    log_id = _next_id(logs)
    for r in records:
        logs.append({
            "id": log_id,
            "session_id": session_id,
            "register_no": str(r.get("Register No", r.get("register_no", ""))).strip(),
            "name": str(r.get("Name", r.get("name", ""))).strip(),
            "department": str(r.get("Department", r.get("department", ""))).strip(),
            "status": str(r.get("Status", r.get("status", "Absent"))).strip()
        })
        log_id += 1
    _write_json(_LOGS_FILE, logs)
    return session_id


def get_all_attendance_sessions(faculty_name: Optional[str] = None, **_kwargs) -> List[Dict[str, Any]]:
    sessions = _read_json(_SESSIONS_FILE)
    if faculty_name and faculty_name.strip():
        f_clean = faculty_name.strip().lower()
        sessions = [s for s in sessions if s.get("faculty_name", "").strip().lower() == f_clean]
    return list(reversed(sessions))


def get_attendance_session_details(
    session_id: int, **_kwargs
) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, str]]]:
    sessions = _read_json(_SESSIONS_FILE)
    session_info = next((s for s in sessions if s.get("id") == session_id), None)
    if not session_info:
        return None, []

    logs = _read_json(_LOGS_FILE)
    records = [
        {
            "Register No": lg.get("register_no", ""),
            "Name": lg.get("name", ""),
            "Department": lg.get("department", ""),
            "Status": lg.get("status", "")
        }
        for lg in logs if lg.get("session_id") == session_id
    ]
    return session_info, records


def delete_attendance_session(session_id: int, **_kwargs) -> bool:
    sessions = _read_json(_SESSIONS_FILE)
    new_sessions = [s for s in sessions if s.get("id") != session_id]
    if len(new_sessions) == len(sessions):
        return False
    _write_json(_SESSIONS_FILE, new_sessions)

    logs = _read_json(_LOGS_FILE)
    _write_json(_LOGS_FILE, [lg for lg in logs if lg.get("session_id") != session_id])
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
    runs = _read_json(_TRAINING_FILE)
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
    _write_json(_TRAINING_FILE, runs)
    return run_id


def get_all_model_training_runs(**_kwargs) -> List[Dict[str, Any]]:
    runs = _read_json(_TRAINING_FILE)
    return list(reversed(runs))


def get_latest_model_training_run(**_kwargs) -> Optional[Dict[str, Any]]:
    runs = _read_json(_TRAINING_FILE)
    return runs[-1] if runs else None


# ── System Settings ──────────────────────────────────────────────────────────

def save_setting(key: str, value: str, **_kwargs) -> None:
    settings = _read_json(_SETTINGS_FILE)
    settings[key] = str(value)
    _write_json(_SETTINGS_FILE, settings)


def get_setting(key: str, default: Optional[str] = None, **_kwargs) -> Optional[str]:
    settings = _read_json(_SETTINGS_FILE)
    return settings.get(key, default)
