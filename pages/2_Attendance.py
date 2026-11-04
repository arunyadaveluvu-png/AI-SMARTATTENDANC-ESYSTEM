import streamlit as st
import cv2
import numpy as np
import pandas as pd
import io
import datetime
import calendar
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import database
import face_utils
import export_utils
import yolo_workflow
import location_utils
import importlib
importlib.reload(database)

st.set_page_config(
    page_title="Attendance Portal – AI Attendance",
    page_icon="📸",
    layout="wide",
    initial_sidebar_state="expanded"
)

database.init_db()

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
html, body, .stApp { background-color: #0f172a !important; color: #f8fafc !important;
    font-family: 'Inter', sans-serif !important; }
[data-testid="stSidebar"] { background-color: #1e293b !important; border-right: 1px solid #334155 !important; }
[data-testid="stMetric"] { background:#1e293b !important; border:1px solid #334155 !important;
    border-top:3px solid #0284c7 !important; border-radius:8px; padding:1rem 1.25rem; }
[data-testid="stMetricValue"] { font-size:1.8rem !important; font-weight:800 !important; color:#38bdf8 !important; }
[data-testid="stMetricLabel"] { font-size:0.82rem !important; color:#cbd5e1 !important;
    font-weight:700; text-transform:uppercase; }
.stTabs [data-baseweb="tab-list"] { background-color:#1e293b !important; padding:6px;
    border-radius:8px; border:1px solid #334155 !important; }
.stTabs [data-baseweb="tab"] { color:#94a3b8 !important; font-weight:600; }
.stTabs [aria-selected="true"] { background-color:#0284c7 !important; color:#fff !important; border-radius:6px; }
.stButton > button { background-color:#0284c7 !important; color:#fff !important;
    font-weight:600 !important; border-radius:6px !important; border:none !important; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="background:linear-gradient(135deg,#15803d,#166534 60%,#0f172a);
    padding:1.5rem 2rem;border-radius:12px;border-top:4px solid #4ade80;margin-bottom:1.5rem;">
  <div style="font-size:1.9rem;font-weight:800;color:#fff;">📸 Classroom Attendance Portal</div>
  <div style="font-size:0.9rem;color:#4ade80;font-weight:600;margin-top:4px;">
    YOLOv8 · InsightFace · GPS Verification · PDF / Excel Export
  </div>
</div>
""", unsafe_allow_html=True)

# ── Session State ─────────────────────────────────────────────────────────────
for key, default in [
    ("faculty_logged_in", False),
    ("faculty_user", None),
    ("cam_snapshots_list", []),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── Helper ────────────────────────────────────────────────────────────────────
def load_image_as_bgr(uploaded_file) -> np.ndarray:
    file_bytes = np.asarray(bytearray(uploaded_file.getvalue()), dtype=np.uint8)
    return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

# ══════════════════════════════════════════════════════════════════════════════
# LOGIN GATE
# ══════════════════════════════════════════════════════════════════════════════
if not st.session_state["faculty_logged_in"]:
    st.markdown("### 🔐 Faculty Login Required")
    st.caption("Default credentials: **faculty** / **password123**")
    col_c = st.columns([1, 2, 1])[1]
    with col_c:
        with st.form("att_login_form"):
            username = st.text_input("Username", value="faculty")
            password = st.text_input("Password", type="password", value="password123")
            login_btn = st.form_submit_button("Login", use_container_width=True)
            if login_btn:
                faculty = database.verify_faculty(username, password)
                if faculty:
                    st.session_state["faculty_logged_in"] = True
                    st.session_state["faculty_user"] = faculty
                    st.rerun()
                else:
                    st.error("Invalid credentials. Try: faculty / password123")
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# LOGGED IN — MAIN PORTAL
# ══════════════════════════════════════════════════════════════════════════════
faculty_name = st.session_state["faculty_user"]["name"]
hdr_col, logout_col = st.columns([4, 1])
with hdr_col:
    st.markdown(f"### 👋 Welcome, **{faculty_name}**")
with logout_col:
    if st.button("Logout", use_container_width=True):
        st.session_state["faculty_logged_in"] = False
        st.session_state["faculty_user"] = None
        st.session_state["cam_snapshots_list"] = []
        st.rerun()

st.markdown("---")

tab_capture, tab_edit, tab_history, tab_students, tab_yolo = st.tabs([
    "📷 Capture Attendance",
    "✏️ Edit Attendance",
    "📅 History & Calendar",
    "👥 Manage Students",
    "🤖 YOLO Studio"
])

YOLO_MODEL  = "yolov8n.pt"
YOLO_CONF   = 0.25
CONF_THRESH = 0.50

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — CAPTURE ATTENDANCE
# ══════════════════════════════════════════════════════════════════════════════
with tab_capture:
    st.subheader("Classroom Photo Capture & AI Processing")

    slot_name = st.text_input("Class / Session / Slot Name *", value="Slot A - Morning Class", key="att_slot_name")

    st.markdown("---")
    location_data = location_utils.render_gps_permission_ui(key_prefix="att_flow")
    st.markdown("---")

    # Webcam notice
    st.info(
        "📷 **Live Webcam:** Works on HTTPS (Streamlit Cloud ✅). "
        "Click **Allow** when the browser asks for camera permission. "
        "If the camera appears blank or blocked, switch to **Upload Photo**."
    )

    input_src = st.radio(
        "Choose Input Method:",
        ["📁 Upload Classroom Photo(s) (up to 10)", "📷 Live Camera Multi-Snapshot (up to 10)"],
        horizontal=True,
        key="att_input_src"
    )

    uploaded_files = []
    if input_src == "📁 Upload Classroom Photo(s) (up to 10)":
        raw_files = st.file_uploader(
            "Upload Classroom Photo(s) — JPG / PNG", type=["jpg", "jpeg", "png"],
            accept_multiple_files=True, key="att_up_files"
        )
        if raw_files:
            uploaded_files = raw_files[:10]
    else:
        c_cnt = len(st.session_state["cam_snapshots_list"])
        c_col1, c_col2 = st.columns([3, 1])
        with c_col1:
            label = f"📷 Take Snapshot #{c_cnt + 1}" if c_cnt < 10 else "✅ Max 10 Snapshots Captured"
            cam_snap = st.camera_input(label, key="att_cam_input")
            if cam_snap is not None and c_cnt < 10:
                prev = st.session_state["cam_snapshots_list"]
                if not prev or prev[-1].getvalue() != cam_snap.getvalue():
                    st.session_state["cam_snapshots_list"].append(cam_snap)
                    st.rerun()
        with c_col2:
            st.metric("Snapshots", f"{c_cnt} / 10")
            if st.session_state["cam_snapshots_list"]:
                if st.button("🗑 Clear Snapshots", use_container_width=True, key="att_clear_snaps"):
                    st.session_state["cam_snapshots_list"] = []
                    st.rerun()
        uploaded_files = st.session_state["cam_snapshots_list"]

    # Preview
    if uploaded_files:
        st.markdown("#### Classroom Photos Preview")
        prev_cols = st.columns(min(len(uploaded_files), 4))
        for idx, f in enumerate(uploaded_files):
            with prev_cols[idx % len(prev_cols)]:
                f_bgr = load_image_as_bgr(f)
                if f_bgr is not None:
                    stamped = location_utils.stamp_gps_on_image(
                        image_bgr=f_bgr, location_data=location_data,
                        slot_name=slot_name.strip(),
                        timestamp_str=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    )
                    st.image(cv2.cvtColor(stamped, cv2.COLOR_BGR2RGB),
                             caption=f"Photo #{idx+1} (GPS stamped)", use_container_width=True)

        if st.button(f"🚀 Process Attendance — {len(uploaded_files)} Photo(s)",
                     use_container_width=True, type="primary", key="att_btn_process"):
            registered_students = database.get_all_students()
            if not registered_students:
                st.warning("No students registered yet. Please register students first.")
            else:
                with st.spinner("Running YOLOv8 detection and face matching…"):
                    union_present = set()
                    annotated_imgs = []
                    total_faces = 0
                    total_yolo = 0
                    yolo_speeds = []
                    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                    for img_file in uploaded_files:
                        img_bgr = load_image_as_bgr(img_file)
                        if img_bgr is None:
                            continue
                        ann_bgr, records, metrics = face_utils.recognize_faces_in_group(
                            group_image_bgr=img_bgr,
                            registered_students=registered_students,
                            threshold=CONF_THRESH,
                            yolo_model_name=YOLO_MODEL,
                            yolo_conf_threshold=YOLO_CONF,
                            use_yolo=True,
                            location_data=location_data,
                            slot_name=slot_name.strip(),
                            timestamp_str=ts
                        )
                        total_faces += metrics.get("total_detected", 0)
                        total_yolo  += metrics.get("yolo_persons_detected", 0)
                        yolo_speeds.append(metrics.get("yolo_speed", {}))
                        annotated_imgs.append(cv2.cvtColor(ann_bgr, cv2.COLOR_BGR2RGB))
                        for r in records:
                            if r["Status"] == "Present":
                                union_present.add(r["Register No"])

                    combined = []
                    for s in registered_students:
                        reg = s["register_no"]
                        is_present = reg in union_present
                        combined.append({
                            "Register No": reg,
                            "Name": s["name"],
                            "Department": s["department"],
                            "Status": "Present" if is_present else "Absent"
                        })
                        st.session_state[f"slide_toggle_{reg}"] = is_present

                    p_count = sum(1 for r in combined if r["Status"] == "Present")
                    a_count = len(combined) - p_count
                    inf_ms_list = [s.get("inference_ms", 0.0) for s in yolo_speeds if "inference_ms" in s]
                    avg_inf = round(float(np.mean(inf_ms_list)), 1) if inf_ms_list else 0.0

                    now_dt   = datetime.datetime.now()
                    date_str = now_dt.strftime("%Y-%m-%d")
                    time_str = now_dt.strftime("%H:%M:%S")
                    loc_lat  = location_data.get("latitude")  if location_data else None
                    loc_lon  = location_data.get("longitude") if location_data else None
                    loc_addr = location_data.get("address")   if location_data else None

                    database.save_attendance_session(
                        slot_name=slot_name.strip(), faculty_name=faculty_name,
                        date_str=date_str, time_str=time_str,
                        total_students=len(registered_students),
                        present_count=p_count, absent_count=a_count,
                        records=combined,
                        location_lat=loc_lat, location_lon=loc_lon, location_address=loc_addr,
                        yolo_model_name=YOLO_MODEL, detection_count=total_yolo
                    )

                    st.session_state.update({
                        "last_annotated_imgs": annotated_imgs,
                        "last_attendance_records": combined,
                        "last_slot_name": slot_name.strip(),
                        "last_process_date": date_str,
                        "last_process_time": time_str,
                        "last_metrics": {
                            "total_registered": len(registered_students),
                            "total_detected": total_faces,
                            "yolo_persons": total_yolo,
                            "present": p_count,
                            "absent": a_count,
                            "yolo_model": YOLO_MODEL,
                            "yolo_inference_ms": avg_inf,
                            "location_lat": loc_lat,
                            "location_lon": loc_lon,
                            "location_address": loc_addr
                        }
                    })
                    st.success(f"✅ Done! {p_count} present, {a_count} absent. Go to **Edit Attendance** tab to review.")

    # Results summary
    if "last_attendance_records" in st.session_state and "last_metrics" in st.session_state:
        st.markdown("---")
        st.subheader("Processing Summary")
        m = st.session_state["last_metrics"]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Registered", m["total_registered"])
        m2.metric("Faces Detected", m["total_detected"])
        m3.metric("Present", m["present"], delta=f"{m['present']/max(m['total_registered'],1)*100:.1f}%")
        m4.metric("Absent",  m["absent"])

        dl1, dl2, dl3, dl4 = st.columns(4)
        dl1.metric("YOLO Model", m.get("yolo_model", YOLO_MODEL))
        dl2.metric("YOLO Persons", m.get("yolo_persons", 0))
        dl3.metric("Inference", f"{m.get('yolo_inference_ms', 0.0)} ms")
        gps_val = "Not Tagged"
        if m.get("location_lat") and m.get("location_lon"):
            gps_val = f"{m['location_lat']:.4f}, {m['location_lon']:.4f}"
        dl4.metric("GPS", gps_val)
        if m.get("location_address"):
            st.caption(f"📍 {m['location_address']}")

        if st.session_state.get("last_annotated_imgs"):
            st.markdown("### Recognition Output")
            imgs = st.session_state["last_annotated_imgs"]
            if len(imgs) == 1:
                st.image(imgs[0], caption="Green=Matched · Red=Unknown · Cyan=YOLO Person", use_container_width=True)
            else:
                cols = st.columns(min(len(imgs), 3))
                for i, img in enumerate(imgs):
                    with cols[i % len(cols)]:
                        st.image(img, caption=f"Frame #{i+1}", use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — EDIT ATTENDANCE
# ══════════════════════════════════════════════════════════════════════════════
with tab_edit:
    st.subheader("Edit Attendance Roster — Slide Toggle Overrides")

    all_db_students = database.get_all_students()
    if not all_db_students:
        st.warning("No students registered. Please go to Registration first.")
    else:
        existing_map = {r["Register No"]: r for r in st.session_state.get("last_attendance_records", [])}
        records = []
        for s in all_db_students:
            reg = s["register_no"]
            records.append(existing_map.get(reg, {
                "Register No": reg, "Name": s["name"],
                "Department": s["department"], "Status": "Absent"
            }))
        st.session_state["last_attendance_records"] = records

        present_recs = [r for r in records if r["Status"] == "Present"]
        absent_recs  = [r for r in records if r["Status"] == "Absent"]

        m1, m2, m3 = st.columns(3)
        m1.metric("Total Class", len(records))
        m2.metric("Present (Toggle ON)", len(present_recs), delta=f"{len(present_recs)/max(len(records),1)*100:.1f}%")
        m3.metric("Absent (Toggle OFF)", len(absent_recs))

        st.markdown("---")
        b1, b2, b3 = st.columns([1, 1, 2])
        with b1:
            if st.button("✅ Mark All Present", use_container_width=True, key="edit_all_p"):
                for r in records:
                    r["Status"] = "Present"
                    st.session_state[f"slide_toggle_{r['Register No']}"] = True
                st.session_state["last_attendance_records"] = records
                st.rerun()
        with b2:
            if st.button("❌ Mark All Absent", use_container_width=True, key="edit_all_a"):
                for r in records:
                    r["Status"] = "Absent"
                    st.session_state[f"slide_toggle_{r['Register No']}"] = False
                st.session_state["last_attendance_records"] = records
                st.rerun()
        with b3:
            search = st.text_input("🔍 Filter student:", placeholder="Name or Reg No…", key="edit_search")

        st.markdown("---")
        # Column headers
        hc = st.columns([0.5, 1.5, 2, 2, 1.5])
        for col, label in zip(hc, ["#", "Reg No", "Name", "Dept", "Toggle Status"]):
            col.markdown(f"**{label}**")
        st.markdown("---")

        filtered = records
        if search.strip():
            q = search.strip().lower()
            filtered = [r for r in records if q in r["Register No"].lower() or q in r["Name"].lower()]

        status_changed = False
        for idx, r in enumerate(filtered):
            reg = r["Register No"]
            c1, c2, c3, c4, c5 = st.columns([0.5, 1.5, 2, 2, 1.5])
            c1.write(f"**{idx+1}**")
            c2.code(reg, language="text")
            c3.write(f"**{r['Name']}**")
            c4.caption(r["Department"])
            with c5:
                val = st.toggle("Present" if r["Status"] == "Present" else "Absent",
                                value=(r["Status"] == "Present"), key=f"slide_toggle_{reg}")
                if val != (r["Status"] == "Present"):
                    r["Status"] = "Present" if val else "Absent"
                    status_changed = True

        if status_changed:
            st.session_state["last_attendance_records"] = records
            st.rerun()

        st.markdown("---")
        st.subheader("📄 Export Reports")
        cur_slot = st.session_state.get("last_slot_name", "Slot A")
        cur_date = st.session_state.get("last_process_date", datetime.datetime.now().strftime("%Y-%m-%d"))
        cur_time = st.session_state.get("last_process_time", datetime.datetime.now().strftime("%H:%M:%S"))
        last_m   = st.session_state.get("last_metrics", {})
        loc_str  = "Not Recorded"
        if last_m.get("location_lat") and last_m.get("location_lon"):
            loc_str = f"{last_m['location_lat']:.5f}, {last_m['location_lon']:.5f}"
            if last_m.get("location_address"):
                loc_str += f" ({last_m['location_address']})"

        df_records = pd.DataFrame(records)
        excel_bytes = export_utils.create_excel_report(
            slot_name=cur_slot, faculty_name=faculty_name, date_str=cur_date, time_str=cur_time,
            total_strength=len(records), present_count=len(present_recs), absent_count=len(absent_recs),
            df_records=df_records, location_str=loc_str, yolo_model_name=YOLO_MODEL
        )
        pdf_bytes = export_utils.create_pdf_report(
            slot_name=cur_slot, faculty_name=faculty_name, date_str=cur_date, time_str=cur_time,
            total_strength=len(records), present_count=len(present_recs), absent_count=len(absent_recs),
            df_records=df_records, annotated_images=st.session_state.get("last_annotated_imgs", []),
            location_str=loc_str, yolo_model_name=YOLO_MODEL
        )
        ts_file = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        dl1, dl2 = st.columns(2)
        with dl1:
            st.download_button("📥 Download PDF Report", data=pdf_bytes,
                               file_name=f"Attendance_{ts_file}.pdf", mime="application/pdf",
                               use_container_width=True, key="edit_dl_pdf")
        with dl2:
            st.download_button("📊 Download Excel Sheet", data=excel_bytes,
                               file_name=f"Attendance_{ts_file}.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                               use_container_width=True, key="edit_dl_excel")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — HISTORY & CALENDAR
# ══════════════════════════════════════════════════════════════════════════════
with tab_history:
    st.subheader("Attendance History & Calendar")
    saved_sessions = database.get_all_attendance_sessions(faculty_name=faculty_name)

    if not saved_sessions:
        st.info(f"No sessions recorded for {faculty_name} yet.")
    else:
        sessions_by_date = {}
        for s in saved_sessions:
            d = s["date_str"]
            sessions_by_date.setdefault(d, []).append(s)

        now_dt = datetime.datetime.now()
        db_years = sorted(set(
            [int(s["date_str"].split("-")[0]) for s in saved_sessions if "-" in s["date_str"]]
            + [now_dt.year]
        ), reverse=True)
        month_names = ["January","February","March","April","May","June",
                       "July","August","September","October","November","December"]

        cy, cm, cs = st.columns([1.5, 1.5, 2])
        with cy:
            sel_year = st.selectbox("Year:", db_years, index=0, key="cal_yr")
        with cm:
            sel_mon_name = st.selectbox("Month:", month_names, index=now_dt.month-1, key="cal_mn")
            sel_mon_num = month_names.index(sel_mon_name) + 1
        with cs:
            st.metric("Total Sessions", len(saved_sessions))

        st.markdown("---")
        st.markdown(f"### 📅 {sel_mon_name} {sel_year}")

        day_names = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
        hdr = st.columns(7)
        for i, d in enumerate(day_names):
            hdr[i].markdown(f"<div style='text-align:center;font-weight:700;color:#38bdf8;font-size:.85rem;'>{d}</div>", unsafe_allow_html=True)

        for week in calendar.monthcalendar(sel_year, sel_mon_num):
            week_cols = st.columns(7)
            for di, day_num in enumerate(week):
                with week_cols[di]:
                    if day_num == 0:
                        st.markdown("<div style='height:44px;'></div>", unsafe_allow_html=True)
                    else:
                        date_fmt = f"{sel_year:04d}-{sel_mon_num:02d}-{day_num:02d}"
                        if date_fmt in sessions_by_date:
                            cnt = len(sessions_by_date[date_fmt])
                            st.markdown(f"<div style='background:#15803d;border:2px solid #22c55e;border-radius:8px;text-align:center;padding:8px 4px;color:#fff;font-weight:700;font-size:.95rem;margin-bottom:2px;'>{day_num}<br><span style='font-size:.65rem;'>{cnt} rec</span></div>", unsafe_allow_html=True)
                            if st.button("View", key=f"cal_{date_fmt}", use_container_width=True):
                                st.session_state["sel_cal_date"] = date_fmt
                                st.rerun()
                        else:
                            st.markdown(f"<div style='background:#7f1d1d;border:2px solid #ef4444;border-radius:8px;text-align:center;padding:8px 4px;color:#fca5a5;font-weight:600;font-size:.95rem;margin-bottom:2px;opacity:.7;'>{day_num}</div>", unsafe_allow_html=True)

        st.markdown("---")
        active_date = st.session_state.get("sel_cal_date")
        if active_date:
            st.markdown(f"### Records for {active_date}")
            for s in sessions_by_date.get(active_date, []):
                s_id = s["id"]
                with st.expander(f"Slot: {s['slot_name']} | {s['time_str']} | Present: {s['present_count']}/{s['total_students']}", expanded=True):
                    info, logs = database.get_attendance_session_details(s_id)
                    if info:
                        c1,c2,c3,c4 = st.columns(4)
                        c1.metric("Date", info["date_str"])
                        c2.metric("Strength", info["total_students"])
                        c3.metric("Present", info["present_count"])
                        c4.metric("Absent",  info["absent_count"])

                        if info.get("location_lat") and info.get("location_lon"):
                            st.info(f"📍 GPS: `{info['location_lat']:.5f}°N, {info['location_lon']:.5f}°E` — {info.get('location_address','')}")

                        df_logs = pd.DataFrame(logs)
                        st.markdown("**Present Students:**")
                        present_df = df_logs[df_logs["Status"] == "Present"].reset_index(drop=True) if not df_logs.empty else pd.DataFrame()
                        if not present_df.empty:
                            present_df.insert(0, "S.No", range(1, len(present_df)+1))
                            st.dataframe(present_df, use_container_width=True, hide_index=True)

                        loc_s = f"{info['location_lat']:.5f}, {info['location_lon']:.5f}" if (info.get('location_lat') and info.get('location_lon')) else "Not Recorded"
                        h_pdf = export_utils.create_pdf_report(
                            slot_name=info["slot_name"], faculty_name=info["faculty_name"],
                            date_str=info["date_str"], time_str=info["time_str"],
                            total_strength=info["total_students"], present_count=info["present_count"],
                            absent_count=info["absent_count"], df_records=df_logs,
                            location_str=loc_s, yolo_model_name=info.get("yolo_model_name","")
                        )
                        h_xl = export_utils.create_excel_report(
                            slot_name=info["slot_name"], faculty_name=info["faculty_name"],
                            date_str=info["date_str"], time_str=info["time_str"],
                            total_strength=info["total_students"], present_count=info["present_count"],
                            absent_count=info["absent_count"], df_records=df_logs,
                            location_str=loc_s, yolo_model_name=info.get("yolo_model_name","")
                        )
                        hc1, hc2, hc3 = st.columns([2,2,1])
                        with hc1:
                            st.download_button(f"📥 PDF ({info['date_str']})", data=h_pdf,
                                file_name=f"Attendance_{info['date_str']}_{s_id}.pdf", mime="application/pdf",
                                use_container_width=True, key=f"h_pdf_{s_id}")
                        with hc2:
                            st.download_button(f"📊 Excel ({info['date_str']})", data=h_xl,
                                file_name=f"Attendance_{info['date_str']}_{s_id}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                use_container_width=True, key=f"h_xl_{s_id}")
                        with hc3:
                            if st.button("🗑 Delete", key=f"h_del_{s_id}", use_container_width=True):
                                database.delete_attendance_session(s_id)
                                st.session_state["sel_cal_date"] = None
                                st.success("Session deleted.")
                                st.rerun()
        else:
            st.info("Click a green date in the calendar to view that day's attendance records.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — MANAGE STUDENTS
# ══════════════════════════════════════════════════════════════════════════════
with tab_students:
    st.subheader("Registered Students Directory")
    all_students = database.get_all_students()
    if not all_students:
        st.warning("No students registered yet.")
    else:
        sm1, sm2 = st.columns(2)
        sm1.metric("Total Registered", len(all_students))
        sm2.metric("Storage", ".local/students.json")

        st.dataframe(pd.DataFrame([
            {"Register No": s["register_no"], "Name": s["name"], "Department": s["department"]}
            for s in all_students
        ]), use_container_width=True, hide_index=True)

        st.markdown("---")
        st.subheader("Edit or Delete a Student")
        student_map = {f"{s['register_no']} — {s['name']}": s for s in all_students}
        sel = st.selectbox("Select student:", list(student_map.keys()), key="mgmt_sel")
        if sel:
            tgt = student_map[sel]
            ec, dc = st.columns([1.2, 1], gap="large")
            with ec:
                st.markdown("#### ✏️ Edit Details")
                with st.form(f"edit_form_{tgt['register_no']}"):
                    new_name = st.text_input("Full Name", value=tgt["name"])
                    dept_opts = ["Computer Science & Engineering","Information Technology",
                                 "Electrical & Electronics","Mechanical Engineering",
                                 "Civil Engineering","Biotechnology","Other"]
                    curr_idx = dept_opts.index(tgt["department"]) if tgt["department"] in dept_opts else len(dept_opts)-1
                    new_dept = st.selectbox("Department", dept_opts, index=curr_idx)
                    if st.form_submit_button("Save Changes", use_container_width=True):
                        if not new_name.strip():
                            st.error("Name cannot be empty.")
                        elif database.update_student(tgt["register_no"], new_name, new_dept):
                            st.success("Updated!")
                            st.rerun()
                        else:
                            st.error("Update failed.")
            with dc:
                st.markdown("#### 🗑 Delete Student")
                st.warning(f"Delete **{tgt['name']}** ({tgt['register_no']})?")
                if st.button("Delete Student", use_container_width=True, key=f"del_{tgt['register_no']}"):
                    if database.delete_student(tgt["register_no"]):
                        st.success("Deleted.")
                        st.rerun()
                    else:
                        st.error("Delete failed.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — YOLO STUDIO
# ══════════════════════════════════════════════════════════════════════════════
with tab_yolo:
    st.subheader("🤖 YOLO Deep-Learning Studio")

    model_info = yolo_workflow.get_yolo_model_info(YOLO_MODEL)
    s1,s2,s3,s4 = st.columns(4)
    s1.metric("Architecture", "YOLOv8 Nano")
    s2.metric("Weights",      model_info.get("model_name", YOLO_MODEL))
    s3.metric("Parameters",   model_info.get("parameters", "3.2M"))
    s4.metric("Device",       model_info.get("device", "CPU").upper())

    st.markdown("---")
    st.markdown("### Live YOLO Inference Test")
    test_file = st.file_uploader("Upload test image for YOLO detection:", type=["jpg","jpeg","png"], key="yolo_test_up")
    if test_file:
        test_bgr = load_image_as_bgr(test_file)
        if test_bgr is not None:
            with st.spinner("Running YOLO detection…"):
                detections, speeds = yolo_workflow.detect_people_yolo(test_bgr, conf_threshold=YOLO_CONF)
                ann = test_bgr.copy()
                for det in detections:
                    b = det["bbox"]
                    cv2.rectangle(ann, (b[0],b[1]),(b[2],b[3]),(0,255,0),2)
                    cv2.putText(ann, f"{det['class_name']} [{det['confidence']:.2f}]",
                                (b[0], max(15, b[1]-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
            img_col, res_col = st.columns([1.5, 1])
            with img_col:
                st.image(cv2.cvtColor(ann, cv2.COLOR_BGR2RGB), caption="YOLO Detection Result", use_container_width=True)
            with res_col:
                st.metric("Detections", len(detections))
                st.metric("Inference", f"{speeds['inference_ms']} ms")
                st.metric("Total Time", f"{speeds['total_ms']} ms")
                if detections:
                    st.dataframe(pd.DataFrame(detections)[["class_name","confidence"]], use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### Model Training")
    st.info("To fine-tune YOLO, provide a YOLO-format `data.yaml` and annotated classroom images. The model weights need to be in the app directory.")
    t1, t2 = st.columns([1.5, 1])
    with t1:
        yaml_path = st.text_input("Dataset YAML path:", value="yolo_dataset/data.yaml", key="yolo_yaml")
        ep_col, bt_col, lr_col = st.columns(3)
        t_epochs = ep_col.slider("Epochs:", 1, 50, 5, key="yolo_ep")
        t_batch  = bt_col.select_slider("Batch:", [2,4,8,16], value=4, key="yolo_bt")
        t_lr     = lr_col.selectbox("LR:", [0.01,0.005,0.001], key="yolo_lr")
        if st.button("🚀 Start Training", use_container_width=True, key="yolo_train_btn"):
            if not os.path.exists(yaml_path):
                st.error(f"Dataset config `{yaml_path}` not found. Create a YOLO dataset first.")
            else:
                prog = st.progress(0, text="Starting…")
                def cb(d):
                    prog.progress(int(min(d.get("progress_pct",0), 100)),
                                  text=f"Epoch {d.get('epochs_trained',0)}/{d.get('total_epochs',t_epochs)}")
                with st.spinner("Training…"):
                    result = yolo_workflow.train_yolo_model(yaml_path, YOLO_MODEL, t_epochs, t_batch, t_lr, cb)
                if result.get("success"):
                    prog.progress(100, text="Done!")
                    st.success("Training complete!")
                    m = result.get("metrics", {})
                    database.save_model_training_run(
                        model_name=YOLO_MODEL, dataset_name=os.path.basename(yaml_path),
                        epochs_trained=m.get("epochs_trained", t_epochs), total_epochs=t_epochs,
                        final_loss=m.get("total_loss"), box_loss=m.get("box_loss"),
                        cls_loss=m.get("cls_loss"), dfl_loss=m.get("dfl_loss"),
                        precision_score=m.get("precision"), recall_score=m.get("recall"),
                        map50=m.get("map50"), map50_95=m.get("map50_95"),
                        training_status="COMPLETED",
                        started_at=m.get("started_at"), completed_at=m.get("completed_at"),
                        notes=f"batch={t_batch}, lr={t_lr}"
                    )
                else:
                    st.error(f"Training failed: {result.get('error')}")
    with t2:
        st.markdown("#### Required Dataset Structure")
        st.code("""
yolo_dataset/
  data.yaml
  images/
    train/   *.jpg
    val/     *.jpg
  labels/
    train/   *.txt
    val/     *.txt
""", language="text")
        st.caption("Each `.txt` label: `<class_id> <x_center> <y_center> <width> <height>` (normalized 0–1)")
