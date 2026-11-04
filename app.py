import streamlit as st
import cv2
import numpy as np
import pandas as pd
import io
import datetime
import calendar
import os
import json
from typing import Optional

import database
import face_utils
import export_utils
import yolo_workflow
import location_utils

# Page Config
st.set_page_config(
    page_title="AI Smart Attendance & Registration System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize database
database.init_db()

# Custom CSS for Professional Dark Executive Theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, .stApp {
        background-color: #0f172a !important;
        color: #f8fafc !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }
    [data-testid="stSidebar"] {
        background-color: #1e293b !important;
        border-right: 1px solid #334155 !important;
    }
    .portal-banner {
        background: linear-gradient(135deg, #0284c7 0%, #0369a1 60%, #0f172a 100%) !important;
        padding: 1.5rem 2rem;
        border-radius: 12px;
        border-top: 4px solid #38bdf8 !important;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px rgba(2, 132, 199, 0.25);
    }
    .portal-banner-title {
        font-size: 2rem;
        font-weight: 800;
        color: #ffffff !important;
        margin: 0;
    }
    .portal-banner-sub {
        font-size: 0.95rem;
        color: #bae6fd !important;
        margin-top: 4px;
        font-weight: 600;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #1e293b !important;
        padding: 6px;
        border-radius: 8px;
        border: 1px solid #334155 !important;
    }
    .stTabs [data-baseweb="tab"] {
        height: 44px;
        background-color: transparent !important;
        border-radius: 6px;
        color: #94a3b8 !important;
        font-weight: 600;
        font-size: 0.9rem;
        padding: 0 18px;
        border: none !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: #0284c7 !important;
        color: #ffffff !important;
    }
    .stTabs [aria-selected="true"] p, .stTabs [aria-selected="true"] span {
        color: #ffffff !important;
    }
    .stButton > button, div[data-testid="stFormSubmitButton"] > button {
        background-color: #0284c7 !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        border: none !important;
        border-radius: 6px !important;
    }
    .stButton > button:hover, div[data-testid="stFormSubmitButton"] > button:hover {
        background-color: #0369a1 !important;
        color: #ffffff !important;
    }
    [data-testid="stMetric"], .card-box {
        background: #1e293b !important;
        border: 1px solid #334155 !important;
        border-top: 3px solid #0284c7 !important;
        border-radius: 8px;
        padding: 1rem 1.25rem;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        font-weight: 800 !important;
        color: #38bdf8 !important;
    }
    div[data-baseweb="input"] {
        background-color: #1e293b !important;
        border-color: #334155 !important;
    }
    div[data-baseweb="input"] input {
        color: #f8fafc !important;
    }
</style>
""", unsafe_allow_html=True)

from PIL import Image

# Helper function
def load_image_as_bgr(uploaded_file) -> Optional[np.ndarray]:
    if uploaded_file is None:
        return None
    try:
        img = Image.open(uploaded_file).convert("RGB")
        rgb = np.array(img)
        return rgb[:, :, ::-1]  # RGB to BGR
    except Exception:
        try:
            file_bytes = np.asarray(bytearray(uploaded_file.getvalue()), dtype=np.uint8)
            return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        except Exception:
            return None

# Session state initialization
if "faculty_logged_in" not in st.session_state:
    st.session_state["faculty_logged_in"] = False
if "faculty_user" not in st.session_state:
    st.session_state["faculty_user"] = None
if "att_snapshots" not in st.session_state:
    st.session_state["att_snapshots"] = []

# Sidebar Navigation
st.sidebar.markdown("## 🎓 AI Attendance System")
active_portal = st.sidebar.radio(
    "Select System Portal:",
    [
        "👤 Portal 1: Registration",
        "📸 Portal 2: Attendance Capture",
        "📅 Portal 3: Attendance History & Reports"
    ],
    index=0
)

# Sidebar System Quick Stats
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 System Status")
st_count = len(database.get_all_students())
sess_count = len(database.get_all_attendance_sessions())
st.sidebar.metric("Registered Students", st_count)
st.sidebar.metric("Saved Sessions", sess_count)

if database.is_supabase_connected():
    st.sidebar.success("🟢 Cloud Sync: Supabase Active")
else:
    st.sidebar.info("💾 Storage: Local JSON")

with st.sidebar.expander("🔄 JSON Sync / Backup", expanded=False):
    st.download_button(
        "📥 Export students.json",
        data=database.export_students_json_bytes(),
        file_name="students.json",
        mime="application/json",
        use_container_width=True,
        key="app_dl_students_json"
    )
    up_sync = st.file_uploader("Upload students.json", type=["json"], key="app_sync_json_up")
    if up_sync is not None:
        added, updated = database.import_students_json_bytes(up_sync.getvalue())
        st.success(f"Synced! Added {added}, Updated {updated}.")
        st.rerun()

# ==============================================================================
# PORTAL 1: REGISTRATION PORTAL
# ==============================================================================
if active_portal == "👤 Portal 1: Registration":
    st.markdown("""
    <div class="portal-banner">
        <div class="portal-banner-title">👤 Student & Faculty Registration Portal</div>
        <div class="portal-banner-sub">Enroll student facial biometric profiles and create faculty access accounts</div>
    </div>
    """, unsafe_allow_html=True)

    tab_stud, tab_fac = st.tabs(["🧑‍🎓 Student Face Registration", "👩‍🏫 Faculty Account Registration"])

    with tab_stud:
        col_form, col_roster = st.columns([1.2, 1], gap="large")

        with col_form:
            st.subheader("Student Details")
            r_no = st.text_input("Register Number *", placeholder="e.g. 24BCA8011", key="reg_rno")
            f_name = st.text_input("Full Name *", placeholder="e.g. Jane Doe", key="reg_name")
            dept = st.selectbox(
                "Department *",
                [
                    "Computer Science & Engineering",
                    "Information Technology",
                    "Electrical & Electronics",
                    "Mechanical Engineering",
                    "Civil Engineering",
                    "Biotechnology",
                    "Other"
                ],
                key="reg_dept"
            )

            st.markdown("---")
            st.markdown("### 📷 Face Photo Capture")
            
            # Camera advice notice
            st.info(
                "💡 **Webcam Tip:** To use your live camera, click **Allow** when your browser prompts for camera permission. "
                "If using a phone or if camera access is blocked, you can use **Upload Photo File**."
            )

            input_mode = st.radio(
                "Choose Face Input Method:",
                ["📷 Live Webcam", "📁 Upload Photo File"],
                horizontal=True,
                key="reg_mode_radio"
            )

            student_photo = None
            if input_mode == "📷 Live Webcam":
                cam_img = st.camera_input("Position face clearly in the camera frame", key="reg_camera_widget")
                if cam_img is not None:
                    student_photo = cam_img
                    st.success("✅ Snapshot captured ready for enrollment!")
            else:
                up_img = st.file_uploader("Upload clear student photo (JPG, PNG)", type=["jpg", "jpeg", "png"], key="reg_file_widget")
                if up_img is not None:
                    student_photo = up_img

            st.markdown("---")
            if st.button("Register Student Profile", use_container_width=True, type="primary", key="btn_register_student"):
                if not r_no.strip() or not f_name.strip():
                    st.error("Please enter both Register Number and Full Name.")
                elif student_photo is None:
                    st.error("Please take a photo with the webcam or upload a photo file.")
                elif database.student_exists(r_no):
                    st.error(f"Student with Register Number '{r_no.strip()}' is already registered!")
                else:
                    with st.spinner("Extracting facial embeddings using InsightFace..."):
                        img_bgr = load_image_as_bgr(student_photo)
                        if img_bgr is None:
                            st.error("Could not decode image file.")
                        else:
                            emb, code, msg = face_utils.extract_single_face_embedding(img_bgr)
                            if code == "NO_FACE":
                                st.error(f"❌ {msg}")
                            elif code == "MULTIPLE_FACES":
                                st.warning(f"⚠️ {msg}")
                            elif code == "SUCCESS":
                                if database.add_student(r_no, f_name, dept, emb):
                                    st.success(f"🎉 Student '{f_name}' ({r_no.strip()}) registered successfully!")
                                    st.rerun()
                                else:
                                    st.error("Database save failed.")
                            else:
                                st.error(f"❌ {msg}")

        with col_roster:
            st.subheader("Registered Students Roster")
            all_st = database.get_all_students()
            st.metric("Total Enrolled Students", len(all_st))
            if all_st:
                df_st = pd.DataFrame([
                    {"Register No": s["register_no"], "Name": s["name"], "Department": s["department"]}
                    for s in all_st
                ])
                st.dataframe(df_st, use_container_width=True, hide_index=True)
            else:
                st.info("No students enrolled yet. Use the form to enroll your first student.")

    with tab_fac:
        fcol1, fcol2 = st.columns([1.2, 1], gap="large")
        with fcol1:
            st.subheader("New Faculty Registration")
            with st.form("fac_reg_form"):
                fac_name = st.text_input("Full Name *", placeholder="Dr. Alan Turing")
                fac_user = st.text_input("Username *", placeholder="alan_turing")
                fac_pass = st.text_input("Password *", type="password")
                fac_confirm = st.text_input("Confirm Password *", type="password")
                fac_submit = st.form_submit_button("Create Faculty Account", use_container_width=True)

            if fac_submit:
                if not fac_name.strip() or not fac_user.strip() or not fac_pass.strip():
                    st.error("All fields are required.")
                elif fac_pass != fac_confirm:
                    st.error("Passwords do not match!")
                elif database.faculty_exists(fac_user):
                    st.error(f"Username '{fac_user.strip()}' already exists.")
                else:
                    if database.add_faculty(fac_user, fac_pass, fac_name):
                        st.success(f"✅ Faculty account created for '{fac_name}'! You can now log into Attendance Capture.")
                    else:
                        st.error("Database error while saving account.")

        with fcol2:
            st.subheader("Faculty Accounts")
            fac_list = database.get_all_faculty()
            st.metric("Total Faculty", len(fac_list))
            if fac_list:
                st.dataframe(pd.DataFrame(fac_list), use_container_width=True, hide_index=True)

# ==============================================================================
# PORTAL 2: ATTENDANCE CAPTURE PORTAL
# ==============================================================================
elif active_portal == "📸 Portal 2: Attendance Capture":
    st.markdown("""
    <div class="portal-banner" style="background: linear-gradient(135deg, #15803d 0%, #166534 60%, #0f172a 100%) !important; border-top: 4px solid #4ade80 !important;">
        <div class="portal-banner-title">📸 Classroom Attendance Capture Portal</div>
        <div class="portal-banner-sub">YOLOv8 Person Detection · InsightFace Biometrics · GPS Stamp · Real-time Reports</div>
    </div>
    """, unsafe_allow_html=True)

    # Authentication Gate
    if not st.session_state["faculty_logged_in"]:
        st.markdown("### 🔐 Faculty Login Required")
        st.caption("Sign in to capture and submit classroom attendance records.")
        
        login_col = st.columns([1, 2, 1])[1]
        with login_col:
            with st.form("fac_login_box"):
                l_user = st.text_input("Faculty Username", value="faculty")
                l_pass = st.text_input("Faculty Password", type="password", value="password123")
                l_btn = st.form_submit_button("Sign In to Attendance Portal", use_container_width=True)

            if l_btn:
                verified = database.verify_faculty(l_user, l_pass)
                if verified:
                    st.session_state["faculty_logged_in"] = True
                    st.session_state["faculty_user"] = verified
                    st.success(f"Welcome back, {verified['name']}!")
                    st.rerun()
                else:
                    st.error("Invalid credentials. Default: username `faculty` / password `password123`")
    else:
        # Logged In View
        active_fac = st.session_state["faculty_user"]["name"]
        h_left, h_right = st.columns([4, 1])
        with h_left:
            st.markdown(f"### Logged in as: **{active_fac}**")
        with h_right:
            if st.button("Sign Out", use_container_width=True):
                st.session_state["faculty_logged_in"] = False
                st.session_state["faculty_user"] = None
                st.session_state["att_snapshots"] = []
                st.rerun()

        st.markdown("---")
        slot_name = st.text_input("Session / Slot Name *", value="Slot A - Morning Lecture", key="att_slot_val")

        # GPS Location capture
        location_data = location_utils.render_gps_permission_ui(key_prefix="app_capture_gps")

        st.markdown("### 📸 Classroom Image Input")
        st.info(
            "💡 **Camera & Upload Option:** You can take photos directly with the **Live Camera** or upload existing classroom group photos."
        )

        att_mode = st.radio(
            "Choose Capture Method:",
            ["📷 Live Camera Snapshot(s)", "📁 Upload Classroom Photo(s)"],
            horizontal=True,
            key="att_capture_mode_radio"
        )

        final_photos = []
        if att_mode == "📷 Live Camera Snapshot(s)":
            snap_col1, snap_col2 = st.columns([3, 1])
            with snap_col1:
                snap_file = st.camera_input("Point camera at classroom / students and snap", key="att_camera_widget")
                if snap_file is not None:
                    # Avoid duplicate appending of the exact same snapshot
                    bytes_val = snap_file.getvalue()
                    if not any(s.getvalue() == bytes_val for s in st.session_state["att_snapshots"]):
                        if len(st.session_state["att_snapshots"]) < 10:
                            st.session_state["att_snapshots"].append(snap_file)
                            st.success(f"Snapshot #{len(st.session_state['att_snapshots'])} added!")
            with snap_col2:
                st.metric("Captured Snapshots", f"{len(st.session_state['att_snapshots'])} / 10")
                if st.session_state["att_snapshots"]:
                    if st.button("Clear Snapshots", use_container_width=True):
                        st.session_state["att_snapshots"] = []
                        st.rerun()
            final_photos = st.session_state["att_snapshots"]
        else:
            up_files = st.file_uploader(
                "Upload Classroom Group Image(s) (1 to 10 files)",
                type=["jpg", "jpeg", "png"],
                accept_multiple_files=True,
                key="att_group_upload_widget"
            )
            if up_files:
                final_photos = up_files[:10]

        # Preview of photos
        if final_photos:
            st.markdown("#### Classroom Photo Preview")
            p_cols = st.columns(min(len(final_photos), 4))
            for i, p in enumerate(final_photos):
                with p_cols[i % len(p_cols)]:
                    p_bgr = load_image_as_bgr(p)
                    if p_bgr is not None:
                        stamped = location_utils.stamp_gps_on_image(
                            image_bgr=p_bgr,
                            location_data=location_data,
                            slot_name=slot_name.strip(),
                            timestamp_str=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        )
                        st.image(cv2.cvtColor(stamped, cv2.COLOR_BGR2RGB), caption=f"Frame #{i+1}", use_container_width=True)

            if st.button(f"🚀 Run AI Facial Attendance on {len(final_photos)} Photo(s)", use_container_width=True, type="primary"):
                students = database.get_all_students()
                if not students:
                    st.warning("No registered students found! Please register students in Portal 1 first.")
                else:
                    with st.spinner("Executing YOLO person localization & InsightFace facial matching..."):
                        present_regs = set()
                        annotated_frames = []
                        total_faces = 0
                        total_yolo = 0
                        yolo_speeds = []
                        capture_ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                        for p in final_photos:
                            p_bgr = load_image_as_bgr(p)
                            if p_bgr is None:
                                continue
                            ann_bgr, recs, mets = face_utils.recognize_faces_in_group(
                                group_image_bgr=p_bgr,
                                registered_students=students,
                                threshold=0.50,
                                yolo_model_name="yolov8n.pt",
                                yolo_conf_threshold=0.25,
                                use_yolo=True,
                                location_data=location_data,
                                slot_name=slot_name.strip(),
                                timestamp_str=capture_ts
                            )
                            total_faces += mets.get("total_detected", 0)
                            total_yolo += mets.get("yolo_persons_detected", 0)
                            yolo_speeds.append(mets.get("yolo_speed", {}))
                            annotated_frames.append(cv2.cvtColor(ann_bgr, cv2.COLOR_BGR2RGB))
                            for r in recs:
                                if r["Status"] == "Present":
                                    present_regs.add(r["Register No"])

                        combined_records = []
                        for s in students:
                            reg = s["register_no"]
                            is_p = reg in present_regs
                            combined_records.append({
                                "Register No": reg,
                                "Name": s["name"],
                                "Department": s["department"],
                                "Status": "Present" if is_p else "Absent"
                            })
                            st.session_state[f"toggle_{reg}"] = is_p

                        p_cnt = len(present_regs)
                        a_cnt = len(students) - p_cnt
                        now_dt = datetime.datetime.now()
                        d_str = now_dt.strftime("%Y-%m-%d")
                        t_str = now_dt.strftime("%H:%M:%S")

                        # Save to database
                        database.save_attendance_session(
                            slot_name=slot_name.strip(),
                            faculty_name=active_fac,
                            date_str=d_str,
                            time_str=t_str,
                            total_students=len(students),
                            present_count=p_cnt,
                            absent_count=a_cnt,
                            records=combined_records,
                            location_lat=location_data.get("latitude") if location_data else None,
                            location_lon=location_data.get("longitude") if location_data else None,
                            location_address=location_data.get("address") if location_data else None,
                            yolo_model_name="yolov8n.pt",
                            detection_count=total_yolo
                        )

                        st.session_state["cur_annotated"] = annotated_frames
                        st.session_state["cur_records"] = combined_records
                        st.session_state["cur_meta"] = {
                            "slot": slot_name.strip(),
                            "date": d_str,
                            "time": t_str,
                            "present": p_cnt,
                            "absent": a_cnt,
                            "total": len(students),
                            "faces": total_faces,
                            "yolo": total_yolo
                        }
                        st.success(f"✅ Attendance recorded! {p_cnt} Present, {a_cnt} Absent.")

        # Show Results if available
        if "cur_records" in st.session_state:
            st.markdown("---")
            st.subheader("Attendance Results & Manual Adjustment")
            meta = st.session_state.get("cur_meta", {})
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Students", meta.get("total", 0))
            m2.metric("Detected Faces", meta.get("faces", 0))
            m3.metric("Present Count", meta.get("present", 0))
            m4.metric("Absent Count", meta.get("absent", 0))

            records = st.session_state["cur_records"]
            df_recs = pd.DataFrame(records)

            st.markdown("#### Present Students List")
            p_df = df_recs[df_recs["Status"] == "Present"]
            if not p_df.empty:
                st.dataframe(p_df, use_container_width=True, hide_index=True)
            else:
                st.warning("No students detected as Present in this batch.")

            # Annotated image viewer
            if st.session_state.get("cur_annotated"):
                st.markdown("#### Annotated Recognition Frames")
                for img in st.session_state["cur_annotated"]:
                    st.image(img, use_container_width=True)

# ==============================================================================
# PORTAL 3: HISTORY & REPORTS
# ==============================================================================
elif active_portal == "📅 Portal 3: Attendance History & Reports":
    st.markdown("""
    <div class="portal-banner" style="background: linear-gradient(135deg, #4338ca 0%, #3730a3 60%, #0f172a 100%) !important; border-top: 4px solid #818cf8 !important;">
        <div class="portal-banner-title">📅 Attendance Session History & Reports</div>
        <div class="portal-banner-sub">Review past sessions, inspect GPS logs, and download official PDF/Excel reports</div>
    </div>
    """, unsafe_allow_html=True)

    sessions = database.get_all_attendance_sessions()
    if not sessions:
        st.info("No attendance sessions have been recorded yet.")
    else:
        st.metric("Total Recorded Sessions", len(sessions))
        st.markdown("---")

        for s in sessions:
            s_id = s.get("id", 0)
            title = f"Slot: {s.get('slot_name')} | Date: {s.get('date_str')} {s.get('time_str')} | Present: {s.get('present_count')}/{s.get('total_students')} | Faculty: {s.get('faculty_name')}"
            with st.expander(title, expanded=False):
                info, logs = database.get_attendance_session_details(s_id)
                if info:
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Date", info.get("date_str"))
                    c2.metric("Total Strength", info.get("total_students"))
                    c3.metric("Present", info.get("present_count"))
                    c4.metric("Absent", info.get("absent_count"))

                    if info.get("location_lat") and info.get("location_lon"):
                        st.info(f"📍 GPS Verified: `{info.get('location_lat'):.5f}° N, {info.get('location_lon'):.5f}° E` — {info.get('location_address', 'Location recorded')}")

                    df_logs = pd.DataFrame(logs)
                    if not df_logs.empty:
                        st.dataframe(df_logs, use_container_width=True, hide_index=True)

                    # Export options
                    loc_str = f"{info.get('location_lat')}, {info.get('location_lon')}" if info.get("location_lat") else "Not Recorded"
                    pdf_bytes = export_utils.create_pdf_report(
                        slot_name=info.get("slot_name", "Slot A"),
                        faculty_name=info.get("faculty_name", "Faculty"),
                        date_str=info.get("date_str", ""),
                        time_str=info.get("time_str", ""),
                        total_strength=info.get("total_students", 0),
                        present_count=info.get("present_count", 0),
                        absent_count=info.get("absent_count", 0),
                        df_records=df_logs,
                        location_str=loc_str,
                        yolo_model_name=info.get("yolo_model_name", "YOLOv8")
                    )
                    excel_bytes = export_utils.create_excel_report(
                        slot_name=info.get("slot_name", "Slot A"),
                        faculty_name=info.get("faculty_name", "Faculty"),
                        date_str=info.get("date_str", ""),
                        time_str=info.get("time_str", ""),
                        total_strength=info.get("total_students", 0),
                        present_count=info.get("present_count", 0),
                        absent_count=info.get("absent_count", 0),
                        df_records=df_logs,
                        location_str=loc_str,
                        yolo_model_name=info.get("yolo_model_name", "YOLOv8")
                    )

                    btn_c1, btn_c2, btn_c3 = st.columns([1, 1, 1])
                    with btn_c1:
                        st.download_button(
                            "📥 Download PDF Report",
                            data=pdf_bytes,
                            file_name=f"Attendance_{info.get('date_str')}_{s_id}.pdf",
                            mime="application/pdf",
                            key=f"pdf_dl_{s_id}"
                        )
                    with btn_c2:
                        st.download_button(
                            "📊 Download Excel Sheet",
                            data=excel_bytes,
                            file_name=f"Attendance_{info.get('date_str')}_{s_id}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            key=f"xl_dl_{s_id}"
                        )
                    with btn_c3:
                        if st.button("🗑 Delete Session Record", key=f"del_sess_{s_id}"):
                            database.delete_attendance_session(s_id)
                            st.success("Session deleted.")
                            st.rerun()
