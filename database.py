"""
database.py — Unified storage for AI Smart Attendance System.
Supports:
1. Supabase Cloud Sync (when configured via st.secrets or env vars) for automatic real-time sync across multiple websites.
2. Local JSON files (students.json, faculty.json, etc.) as the reliable offline / standalone storage engine.
3. 1-Click JSON Export & Import for instant syncing between different deployed websites without any setup.
"""

import os
import json
import pickle
import base64
import datetime
import requests
from typing import List, Dict, Tuple, Optional, Any

try:
    import streamlit as st
except ImportError:
    st = None

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

# Direct JSON files in root directory
_STUDENTS_FILE = os.path.join(DATA_DIR, "students.json")
_FACULTY_FILE = os.path.join(DATA_DIR, "faculty.json")
_SESSIONS_FILE = os.path.join(DATA_DIR, "attendance_sessions.json")
_LOGS_FILE = os.path.join(DATA_DIR, "attendance_logs.json")
_TRAINING_FILE = os.path.join(DATA_DIR, "model_training_history.json")
_SETTINGS_FILE = os.path.join(DATA_DIR, "settings.json")


# ── Supabase Configuration ──────────────────────────────────────────────────

DEFAULT_SUPABASE_URL = "https://oittgqjtssnutcrrnjqt.supabase.co"
DEFAULT_SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im9pdHRncWp0c3NudXRjcnJuanF0Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3ODU4Mzc1MDksImV4cCI6MjEwMTQxMzUwOX0.9canKPOE-zCklUvPvaDVOYK4rwJF63acmNPqPF8swSM"


def get_supabase_config() -> Tuple[Optional[str], Optional[Dict[str, str]]]:
    """Retrieve Supabase URL and headers using environment, Streamlit Secrets, or default."""
    url = os.environ.get("SUPABASE_URL") or os.environ.get("supabase_url")
    key = os.environ.get("SUPABASE_KEY") or os.environ.get("supabase_key")

    if st and hasattr(st, "secrets"):
        try:
            for k in ["SUPABASE_URL", "supabase_url", "URL", "url"]:
                if not url and k in st.secrets:
                    url = str(st.secrets[k])
            for k in ["SUPABASE_KEY", "supabase_key", "KEY", "key", "anon_key"]:
                if not key and k in st.secrets:
                    key = str(st.secrets[k])
            if (not url or not key) and "supabase" in st.secrets:
                sub = st.secrets["supabase"]
                if hasattr(sub, "get"):
                    if not url:
                        url = sub.get("url") or sub.get("SUPABASE_URL")
                    if not key:
                        key = sub.get("key") or sub.get("SUPABASE_KEY")
        except Exception:
            pass

    if not url:
        url = DEFAULT_SUPABASE_URL
    if not key:
        key = DEFAULT_SUPABASE_KEY

    if url and key:
        url = str(url).strip().rstrip("/")
        key = str(key).strip()
        headers = {
            "apikey": key,
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "Prefer": "return=representation"
        }
        return url, headers
    return None, None


_SUPABASE_STATUS_CACHE: Optional[bool] = None

def is_supabase_connected() -> bool:
    """Test if Supabase cloud is reachable with a lightweight 2s timeout check."""
    global _SUPABASE_STATUS_CACHE
    url, headers = get_supabase_config()
    if not url or not headers:
        return False
    try:
        r = requests.get(f"{url}/rest/v1/", headers=headers, timeout=2.5)
        connected = (r.status_code == 200)
        _SUPABASE_STATUS_CACHE = connected
        return connected
    except Exception:
        _SUPABASE_STATUS_CACHE = False
        return False


# ── File I/O Helpers ────────────────────────────────────────────────────────

def _read_json(path: str) -> Any:
    """Read a JSON file safely."""
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
    except Exception:
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error writing to {path}: {e}")


def _next_id(items: List[Dict]) -> int:
    if not items:
        return 1
    return max(r.get("id", 0) for r in items) + 1


def init_db(**_kwargs) -> None:
    """Seed default faculty and ensure files exist."""
    faculty = _read_json(_FACULTY_FILE)
    if not any(f.get("username") == "faculty" for f in faculty):
        faculty.append({
            "username": "faculty",
            "password": "password123",
            "name": "Faculty Admin"
        })
        _write_json(_FACULTY_FILE, faculty)


# ── Students ────────────────────────────────────────────────────────────────

def add_student(register_no: str, name: str, department: str, embedding: Any, **_kwargs) -> bool:
    """Save student locally and attempt sync to Supabase if connected."""
    reg_clean = register_no.strip()
    b64_emb = ""
    if embedding is not None:
        try:
            b64_emb = base64.b64encode(pickle.dumps(embedding)).decode("utf-8")
        except Exception:
            b64_emb = ""

    payload = {
        "register_no": reg_clean,
        "name": name.strip(),
        "department": department.strip(),
        "embedding": b64_emb
    }

    # 1. Save to Supabase Cloud if reachable
    url, headers = get_supabase_config()
    if url and headers and _SUPABASE_STATUS_CACHE is not False:
        try:
            res = requests.post(f"{url}/rest/v1/students", json=payload, headers=headers, timeout=3)
            if res.status_code in (200, 201):
                pass
        except Exception:
            pass

    # 2. Save to local JSON
    students = _read_json(_STUDENTS_FILE)
    if any(s.get("register_no", "").strip().lower() == reg_clean.lower() for s in students):
        return False
    students.append(payload)
    _write_json(_STUDENTS_FILE, students)
    return True


def student_exists(register_no: str, **_kwargs) -> bool:
    reg_clean = register_no.strip().lower()

    # Check Supabase first if available
    url, headers = get_supabase_config()
    if url and headers and _SUPABASE_STATUS_CACHE is not False:
        try:
            res = requests.get(f"{url}/rest/v1/students?register_no=eq.{reg_clean}&select=register_no", headers=headers, timeout=2)
            if res.status_code == 200 and len(res.json()) > 0:
                return True
        except Exception:
            pass

    students = _read_json(_STUDENTS_FILE)
    return any(s.get("register_no", "").strip().lower() == reg_clean for s in students)


def get_all_students(**_kwargs) -> List[Dict[str, Any]]:
    """Retrieve students from Supabase (if connected) or local JSON file."""
    # Attempt Supabase cloud fetch first
    url, headers = get_supabase_config()
    if url and headers and _SUPABASE_STATUS_CACHE is not False:
        try:
            res = requests.get(f"{url}/rest/v1/students?select=*&order=register_no.asc", headers=headers, timeout=3)
            if res.status_code == 200:
                cloud_data = res.json()
                if cloud_data:
                    # Update local cache
                    _write_json(_STUDENTS_FILE, cloud_data)
                    result = []
                    for s in cloud_data:
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
        except Exception:
            pass

    # Fallback to local JSON
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
    reg_clean = register_no.strip().lower()

    url, headers = get_supabase_config()
    if url and headers and _SUPABASE_STATUS_CACHE is not False:
        try:
            requests.patch(f"{url}/rest/v1/students?register_no=eq.{reg_clean}",
                           json={"name": name.strip(), "department": department.strip()},
                           headers=headers, timeout=3)
        except Exception:
            pass

    students = _read_json(_STUDENTS_FILE)
    for s in students:
        if s.get("register_no", "").strip().lower() == reg_clean:
            s["name"] = name.strip()
            s["department"] = department.strip()
            _write_json(_STUDENTS_FILE, students)
            return True
    return False


def delete_student(register_no: str, **_kwargs) -> bool:
    reg_clean = register_no.strip().lower()

    url, headers = get_supabase_config()
    if url and headers and _SUPABASE_STATUS_CACHE is not False:
        try:
            requests.delete(f"{url}/rest/v1/students?register_no=eq.{reg_clean}", headers=headers, timeout=3)
        except Exception:
            pass

    students = _read_json(_STUDENTS_FILE)
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

    payload = {"username": username.strip(), "password": password.strip(), "name": name.strip()}
    url, headers = get_supabase_config()
    if url and headers and _SUPABASE_STATUS_CACHE is not False:
        try:
            requests.post(f"{url}/rest/v1/faculty", json=payload, headers=headers, timeout=3)
        except Exception:
            pass

    faculty.append(payload)
    _write_json(_FACULTY_FILE, faculty)
    return True


def faculty_exists(username: str, **_kwargs) -> bool:
    faculty = _read_json(_FACULTY_FILE)
    u_clean = username.strip().lower()
    return any(f.get("username", "").strip().lower() == u_clean for f in faculty)


def verify_faculty(username: str, password: str, **_kwargs) -> Optional[Dict[str, Any]]:
    u = username.strip().lower()
    p = password.strip()

    # Supabase check
    url, headers = get_supabase_config()
    if url and headers and _SUPABASE_STATUS_CACHE is not False:
        try:
            res = requests.get(f"{url}/rest/v1/faculty?username=eq.{u}&password=eq.{p}&select=username,name", headers=headers, timeout=3)
            if res.status_code == 200 and len(res.json()) > 0:
                row = res.json()[0]
                return {"username": row.get("username"), "name": row.get("name")}
        except Exception:
            pass

    # Local check
    faculty = _read_json(_FACULTY_FILE)
    for f in faculty:
        if f.get("username", "").strip().lower() == u and f.get("password") == p:
            return {"username": f.get("username"), "name": f.get("name")}

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

    url, headers = get_supabase_config()
    if url and headers and _SUPABASE_STATUS_CACHE is not False:
        try:
            r = requests.post(f"{url}/rest/v1/attendance_sessions", json=session, headers=headers, timeout=3)
            if r.status_code in (200, 201) and len(r.json()) > 0:
                cloud_id = r.json()[0].get("id")
                if cloud_id:
                    session["id"] = cloud_id
                    session_id = cloud_id
        except Exception:
            pass

    sessions.append(session)
    _write_json(_SESSIONS_FILE, sessions)

    log_id = _next_id(logs)
    new_logs = []
    for r in records:
        log_entry = {
            "id": log_id,
            "session_id": session_id,
            "register_no": str(r.get("Register No", r.get("register_no", ""))).strip(),
            "name": str(r.get("Name", r.get("name", ""))).strip(),
            "department": str(r.get("Department", r.get("department", ""))).strip(),
            "status": str(r.get("Status", r.get("status", "Absent"))).strip()
        }
        logs.append(log_entry)
        new_logs.append(log_entry)
        log_id += 1

    if url and headers and _SUPABASE_STATUS_CACHE is not False and new_logs:
        try:
            requests.post(f"{url}/rest/v1/attendance_logs", json=new_logs, headers=headers, timeout=3)
        except Exception:
            pass

    _write_json(_LOGS_FILE, logs)
    return session_id


def get_all_attendance_sessions(faculty_name: Optional[str] = None, **_kwargs) -> List[Dict[str, Any]]:
    url, headers = get_supabase_config()
    if url and headers and _SUPABASE_STATUS_CACHE is not False:
        try:
            endpoint = f"{url}/rest/v1/attendance_sessions?select=*&order=id.desc"
            if faculty_name and faculty_name.strip():
                endpoint += f"&faculty_name=eq.{faculty_name.strip()}"
            res = requests.get(endpoint, headers=headers, timeout=3)
            if res.status_code == 200 and res.json():
                return res.json()
        except Exception:
            pass

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
    url, headers = get_supabase_config()
    if url and headers and _SUPABASE_STATUS_CACHE is not False:
        try:
            requests.delete(f"{url}/rest/v1/attendance_logs?session_id=eq.{session_id}", headers=headers, timeout=3)
            requests.delete(f"{url}/rest/v1/attendance_sessions?id=eq.{session_id}", headers=headers, timeout=3)
        except Exception:
            pass

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


# ── JSON Export / Import Cross-Website Sync ─────────────────────────────────

def export_students_json_bytes() -> bytes:
    """Export all student records into a downloadable JSON file byte array."""
    students = _read_json(_STUDENTS_FILE)
    return json.dumps(students, indent=2, ensure_ascii=False).encode("utf-8")


def import_students_json_bytes(raw_bytes: bytes) -> Tuple[int, int]:
    """
    Import students from an uploaded JSON file into database.
    Returns (added_count, updated_count).
    """
    try:
        incoming = json.loads(raw_bytes.decode("utf-8"))
        if not isinstance(incoming, list):
            return 0, 0
    except Exception:
        return 0, 0

    existing = _read_json(_STUDENTS_FILE)
    existing_map = {s.get("register_no", "").strip().lower(): s for s in existing}

    added = 0
    updated = 0
    for item in incoming:
        reg = item.get("register_no", "").strip()
        if not reg:
            continue
        reg_key = reg.lower()
        if reg_key in existing_map:
            existing_map[reg_key]["name"] = item.get("name", existing_map[reg_key].get("name", ""))
            existing_map[reg_key]["department"] = item.get("department", existing_map[reg_key].get("department", ""))
            if item.get("embedding"):
                existing_map[reg_key]["embedding"] = item["embedding"]
            updated += 1
        else:
            existing_map[reg_key] = item
            added += 1

    merged = list(existing_map.values())
    _write_json(_STUDENTS_FILE, merged)

    # Sync to Supabase if connected
    url, headers = get_supabase_config()
    if url and headers and _SUPABASE_STATUS_CACHE is not False:
        for s in merged:
            try:
                requests.post(f"{url}/rest/v1/students", json=s, headers=headers, timeout=2)
            except Exception:
                pass

    return added, updated
