import streamlit as st
import cv2
import numpy as np
import pandas as pd
import io
import datetime
import calendar
import os
import json

import database
import importlib
importlib.reload(database)
import face_utils
import export_utils
import yolo_workflow
import location_utils

# Default face recognition confidence threshold
DEFAULT_CONFIDENCE_THRESHOLD = 0.50

# Page Configuration
st.set_page_config(
    page_title="Faculty Attendance Management System",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, .stApp {
        background-color: #0f172a !important;
        color: #f8fafc !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }
    .stSidebar {
        background-color: #1e293b !important;
        border-right: 1px solid #334155 !important;
    }

    .main-header {
        font-size: 2.1rem;
        font-weight: 800;
        color: #38bdf8 !important;
        margin-bottom: 0.25rem;
        letter-spacing: -0.02em;
    }
    .sub-header {
        font-size: 0.95rem;
        color: #94a3b8 !important;
        margin-bottom: 1.5rem;
    }

    /* Executive Metric Card */
    [data-testid="stMetric"], .card-box {
        background: #1e293b !important;
        border: 1px solid #334155 !important;
        border-top: 3px solid #0284c7 !important;
        border-radius: 8px;
        padding: 1rem 1.25rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2) !important;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        font-weight: 800 !important;
        color: #38bdf8 !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.82rem !important;
        color: #cbd5e1 !important;
        font-weight: 700;
        text-transform: uppercase;
    }

    /* Deep Learning Metric Badges */
    .dl-badge {
        background: rgba(14, 165, 233, 0.1);
        border: 1px solid rgba(56, 189, 248, 0.3);
        border-radius: 6px;
        padding: 8px 12px;
        margin: 4px 0;
    }

    /* GPS Location Cards */
    .gps-card-granted {
        background: rgba(34, 197, 94, 0.1);
        border: 1px solid #22c55e;
        border-radius: 6px;
        padding: 10px 14px;
        margin-bottom: 12px;
    }
    .gps-card-denied {
        background: rgba(234, 179, 8, 0.1);
        border: 1px solid #eab308;
        border-radius: 6px;
        padding: 10px 14px;
        margin-bottom: 12px;
    }

    /* Clean Professional Tabs */
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
        padding: 0 16px;
        border: none !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: #0284c7 !important;
        color: #ffffff !important;
    }
    .stTabs [aria-selected="true"] p, .stTabs [aria-selected="true"] span {
        color: #ffffff !important;
    }

    /* Professional Executive Buttons */
    .stButton button, .stDownloadButton button, div[data-testid="stFormSubmitButton"] > button {
        background-color: #0284c7 !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 0.55rem 1.3rem !important;
        box-shadow: 0 2px 8px rgba(2, 132, 199, 0.3) !important;
        transition: all 0.2s ease-in-out;
    }
    .stButton button:hover, .stDownloadButton button:hover, div[data-testid="stFormSubmitButton"] > button:hover {
        background-color: #0369a1 !important;
        box-shadow: 0 4px 14px rgba(56, 189, 248, 0.4) !important;
    }

    /* Inputs Styling */
    input, select, textarea, [data-baseweb="select"] {
        border-radius: 6px !important;
        border: 1px solid #334155 !important;
        background-color: #1e293b !important;
        color: #f8fafc !important;
    }

    /* Mobile Responsive Adjustments */
    @media (max-width: 768px) {
        .main-header {font-size: 1.6rem;}
        .sub-header {font-size: 0.85rem;}
        .stTabs [data-baseweb="tab"] {font-size: 0.72rem; padding: 0 10px; height: 36px;}
        .stButton button, .stDownloadButton button, div[data-testid="stFormSubmitButton"] > button {padding: 0.45rem 1rem; font-size: 0.85rem;}
        .stMetricValue {font-size: 1.4rem !important;}
        .stMetricLabel {font-size: 0.7rem !important;}
        .stButton button {min-width: 30px; font-size: 0.85rem;}
    }
    .stButton button[data-key^="cal_btn_"] {
        background-color: #15803d !important;
        color: #ffffff !important;
    }
    .stButton button[data-key^="cal_btn_dis_"] {
        background-color: #7f1d1d !important;
        color: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Database Connection
database.init_db()

# Session State Initializations
if "faculty_logged_in" not in st.session_state:
    st.session_state["faculty_logged_in"] = False
if "faculty_user" not in st.session_state:
    st.session_state["faculty_user"] = None
if "cam_snapshots_list" not in st.session_state:
    st.session_state["cam_snapshots_list"] = []

# Sidebar Navigation Panel
st.sidebar.title("Faculty Portal")
st.sidebar.markdown("**AI Facial Attendance & YOLO Deep-Learning Studio**")
st.sidebar.markdown("---")
st.sidebar.info(
    "Connected Database: SQLite (attendance_system.db)\n\n"
    "Shared Central Database for Student Profiles, Attendance Logs, and Deep Learning Metrics."
)

def load_image_as_bgr(uploaded_file) -> np.ndarray:
    bytes_data = uploaded_file.getvalue()
    file_bytes = np.asarray(bytearray(bytes_data), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return img_bgr

# ==============================================================================
# FACULTY AUTHENTICATION SCREEN
# ==============================================================================
if not st.session_state["faculty_logged_in"]:
    st.markdown('<div class="main-header">Faculty Attendance Management Portal</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Faculty authentication required to access attendance and YOLO deep-learning features.</div>', unsafe_allow_html=True)
    
    col_center = st.columns([1, 2, 1])[1]
    with col_center:
        st.markdown("### Faculty Login")
        with st.form("faculty_login_form"):
            username = st.text_input("Username", value="faculty")
            password = st.text_input("Password", type="password", value="password123")
            login_btn = st.form_submit_button("Login to Attendance Portal", use_container_width=True)
            
            if login_btn:
                faculty = database.verify_faculty(username, password)
                if faculty:
                    st.session_state["faculty_logged_in"] = True
                    st.session_state["faculty_user"] = faculty
                    st.success(f"Authentication successful. Welcome, {faculty['name']}.")
                    st.rerun()
                else:
                    st.error("Invalid Username or Password. (Default: faculty / password123).")

# ==============================================================================
# AUTHENTICATED FACULTY DASHBOARD
# ==============================================================================
else:
    faculty_name = st.session_state["faculty_user"]["name"] if st.session_state["faculty_user"] else "Faculty Admin"
    
    header_col, logout_col = st.columns([4, 1])
    with header_col:
        st.markdown(f"### Welcome, **{faculty_name}**")
    with logout_col:
        if st.button("Logout", use_container_width=True):
            st.session_state["faculty_logged_in"] = False
            st.session_state["faculty_user"] = None
            st.rerun()

    with st.sidebar:
        st.markdown("### Database Connection")
        st.success("Central Database Connected")
            
        registered_cnt = len(database.get_all_students())
        st.metric("Total Registered Students", registered_cnt)
        if st.button("Sync Registered Students", use_container_width=True, key="att_sync_db_btn"):
            st.toast("Database synced!")
            st.rerun()

    tab_att_capture, tab_att_edit, tab_att_history, tab_students_mgmt, tab_yolo_studio = st.tabs([
        "Capture Attendance",
        "Edit Attendance",
        "All-Time Attendance History",
        "Manage Registered Students",
        "YOLO Deep-Learning Studio"
    ])

    # ==========================================================================
    # TAB 1: CAPTURE ATTENDANCE ONLY (FRONT PAGE CAPTURE & PROCESS)
    # ==========================================================================
    with tab_att_capture:
        st.subheader("Classroom Attendance Photo Capture & Automated Processing")
        st.markdown("Upload classroom group photos or take camera snapshots to process attendance with **YOLO deep-learning** detection.")
        
        slot_name = st.text_input("Class / Session / Slot Name *", value="Slot A - Morning Class", key="att_slot_name")
        confidence_threshold = DEFAULT_CONFIDENCE_THRESHOLD

        # Automated YOLO Deep-Learning Detection Engine (Settings hidden from UI per user configuration)
        active_yolo_model = "yolov8n.pt"
        yolo_conf = 0.25
        use_yolo_engine = True

        # --- GPS Geolocation Permission & Capture Flow ---
        st.markdown("---")
        location_data = location_utils.render_gps_permission_ui(key_prefix="att_flow")
        st.markdown("---")

        input_source = st.radio(
            "Select Photo Input Method:",
            ["Upload Classroom Photo(s) (1 to 10)", "Live Camera Snapshot"],
            horizontal=True,
            key="att_input_src"
        )
        
        uploaded_files = []
        if input_source == "Upload Classroom Photo(s) (1 to 10)":
            raw_files = st.file_uploader("Upload Classroom Image(s) (JPG, PNG)", type=["jpg", "jpeg", "png"], accept_multiple_files=True, key="att_up_files")
            if raw_files:
                uploaded_files = raw_files[:10]
        else:
            c_cnt = len(st.session_state["cam_snapshots_list"])
            c_col1, c_col2 = st.columns([3, 1])
            with c_col1:
                cam_snap = st.camera_input(f"Take Snapshot #{c_cnt + 1}" if c_cnt < 10 else "Max 10 Snapshots Captured", key="att_cam_input")
                if cam_snap is not None and c_cnt < 10:
                    if not st.session_state["cam_snapshots_list"] or st.session_state["cam_snapshots_list"][-1].getvalue() != cam_snap.getvalue():
                        st.session_state["cam_snapshots_list"].append(cam_snap)
                        st.rerun()
            with c_col2:
                st.metric("Captured Snapshots", f"{c_cnt} / 10")
                if st.session_state["cam_snapshots_list"]:
                    if st.button("Clear Snapshots", use_container_width=True, key="att_clear_snaps"):
                        st.session_state["cam_snapshots_list"] = []
                        st.rerun()
            uploaded_files = st.session_state["cam_snapshots_list"]

        # Display captured photos with GPS watermark visibly stamped onto the photo
        if uploaded_files:
            st.markdown("#### Classroom Photos Preview & GPS Watermarked Frames")
            preview_cols = st.columns(min(len(uploaded_files), 4))
            for idx, f in enumerate(uploaded_files):
                with preview_cols[idx % len(preview_cols)]:
                    f_bgr = load_image_as_bgr(f)
                    if f_bgr is not None:
                        stamped_preview_bgr = location_utils.stamp_gps_on_image(
                            image_bgr=f_bgr,
                            location_data=location_data,
                            slot_name=slot_name.strip(),
                            timestamp_str=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        )
                        stamped_preview_rgb = cv2.cvtColor(stamped_preview_bgr, cv2.COLOR_BGR2RGB)
                        st.image(stamped_preview_rgb, caption=f"Classroom Photo #{idx + 1} (GPS Watermarked)", use_container_width=True)
                    else:
                        st.image(f, caption=f"Classroom Frame #{idx + 1}", use_container_width=True)

            if st.button(f"Process Attendance Across {len(uploaded_files)} Photo(s)", use_container_width=True, type="primary", key="att_btn_proc"):
                registered_students = database.get_all_students()
                if len(registered_students) == 0:
                    st.warning("No students registered in database. Please register students first.")
                else:
                    with st.spinner(f"Running YOLO deep-learning detection and facial embedding matching across {len(uploaded_files)} frame(s)..."):
                        union_present_regs = set()
                        annotated_images = []
                        total_faces_detected_count = 0
                        total_yolo_persons_count = 0
                        yolo_speed_list = []
                        now_capture_ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        
                        for img_file in uploaded_files:
                            img_bgr = load_image_as_bgr(img_file)
                            if img_bgr is not None:
                                annotated_bgr, records, metrics = face_utils.recognize_faces_in_group(
                                    group_image_bgr=img_bgr,
                                    registered_students=registered_students,
                                    threshold=confidence_threshold,
                                    yolo_model_name=active_yolo_model,
                                    yolo_conf_threshold=yolo_conf,
                                    use_yolo=use_yolo_engine,
                                    location_data=location_data,
                                    slot_name=slot_name.strip(),
                                    timestamp_str=now_capture_ts
                                )
                                total_faces_detected_count += metrics.get("total_detected", 0)
                                total_yolo_persons_count += metrics.get("yolo_persons_detected", 0)
                                yolo_speed_list.append(metrics.get("yolo_speed", {}))
                                
                                annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
                                annotated_images.append(annotated_rgb)
                                for r in records:
                                    if r["Status"] == "Present":
                                        union_present_regs.add(r["Register No"])
                        
                        combined_records = []

                        for s in registered_students:
                            reg = s["register_no"]
                            is_present = reg in union_present_regs
                            combined_records.append({
                                "Register No": reg,
                                "Name": s["name"],
                                "Department": s["department"],
                                "Status": "Present" if is_present else "Absent"
                            })
                            st.session_state[f"slide_toggle_{reg}"] = is_present
                            
                        p_count = len(union_present_regs)
                        a_count = len(registered_students) - p_count
                        
                        # Calculate genuine average YOLO inference latency across frames
                        avg_inference_ms = 0.0
                        if yolo_speed_list:
                            inf_times = [s.get("inference_ms", 0.0) for s in yolo_speed_list if "inference_ms" in s]
                            if inf_times:
                                avg_inference_ms = round(float(np.mean(inf_times)), 1)
                        
                        now_dt = datetime.datetime.now()
                        date_str = now_dt.strftime("%Y-%m-%d")
                        time_str = now_dt.strftime("%H:%M:%S")

                        # Save session with genuine location and YOLO metadata
                        loc_lat = location_data.get("latitude") if location_data else None
                        loc_lon = location_data.get("longitude") if location_data else None
                        loc_addr = location_data.get("address") if location_data else None
                        
                        session_id = database.save_attendance_session(
                            slot_name=slot_name.strip(),
                            faculty_name=faculty_name,
                            date_str=date_str,
                            time_str=time_str,
                            total_students=len(registered_students),
                            present_count=p_count,
                            absent_count=a_count,
                            records=combined_records,
                            location_lat=loc_lat,
                            location_lon=loc_lon,
                            location_address=loc_addr,
                            yolo_model_name=active_yolo_model if use_yolo_engine else "InsightFace Only",
                            detection_count=total_yolo_persons_count
                        )
                        
                        st.session_state["last_session_id"] = session_id
                        st.session_state["last_annotated_imgs"] = annotated_images
                        st.session_state["last_attendance_records"] = combined_records
                        st.session_state["last_metrics"] = {
                            "total_registered": len(registered_students),
                            "total_detected": total_faces_detected_count,
                            "yolo_persons": total_yolo_persons_count,
                            "present": p_count,
                            "absent": a_count,
                            "yolo_model": active_yolo_model if use_yolo_engine else "InsightFace Only",
                            "yolo_inference_ms": avg_inference_ms,
                            "location_lat": loc_lat,
                            "location_lon": loc_lon,
                            "location_address": loc_addr
                        }
                        st.session_state["last_slot_name"] = slot_name.strip()
                        st.session_state["last_process_date"] = date_str
                        st.session_state["last_process_time"] = time_str

                        st.success(f"Attendance processed successfully! Identified {p_count} present student(s). Switch to 'Edit Attendance' tab to view or adjust roster.")

        if "last_attendance_records" in st.session_state and "last_metrics" in st.session_state:
            st.markdown("---")
            st.subheader("Attendance Processing Results Summary")
            metrics = st.session_state.get("last_metrics", {})
            records = st.session_state.get("last_attendance_records", [])
            present_df = pd.DataFrame([r for r in records if r["Status"] == "Present"]) if records else pd.DataFrame()

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Registered Students", metrics["total_registered"])
            m2.metric("Detected Faces", metrics["total_detected"])
            m3.metric("Present Count", metrics["present"], delta=f"{(metrics['present']/max(metrics['total_registered'],1))*100:.1f}%")
            m4.metric("Absent Count", metrics["absent"])

            # Deep Learning & GPS Metadata Card
            st.markdown("#### 🤖 Deep-Learning Inference & Location Telemetry")
            dl_col1, dl_col2, dl_col3, dl_col4 = st.columns(4)
            with dl_col1:
                st.metric("YOLO Model", metrics.get("yolo_model", "yolov8n.pt"))
            with dl_col2:
                st.metric("YOLO Person Detections", metrics.get("yolo_persons", 0))
            with dl_col3:
                st.metric("Inference Latency", f"{metrics.get('yolo_inference_ms', 0.0)} ms")
            with dl_col4:
                loc_txt = "Not Tagged"
                if metrics.get("location_lat") is not None and metrics.get("location_lon") is not None:
                    loc_txt = f"{metrics['location_lat']:.4f}, {metrics['location_lon']:.4f}"
                st.metric("GPS Location", loc_txt)

            if metrics.get("location_address"):
                st.caption(f"📍 **Verified Location:** {metrics['location_address']}")

            if "last_annotated_imgs" in st.session_state and st.session_state["last_annotated_imgs"]:
                st.markdown("### Facial Identification Visual Recognition")
                imgs = st.session_state["last_annotated_imgs"]
                if len(imgs) == 1:
                    st.image(imgs[0], caption="Green = Matched Student | Red = Unknown Face | Cyan = YOLO Detected Person", use_container_width=True)
                else:
                    cols = st.columns(min(len(imgs), 3))
                    for idx, img in enumerate(imgs):
                        with cols[idx % len(cols)]:
                            st.image(img, caption=f"Classroom Photo #{idx+1}", use_container_width=True)

            st.markdown("### Present Students Identified")
            if not present_df.empty:
                disp_df = present_df[["Register No", "Name", "Department", "Status"]].copy()
                disp_df.insert(0, "S.No", range(1, len(disp_df) + 1))
                st.dataframe(disp_df, use_container_width=True, hide_index=True)
            else:
                st.info("No present students identified in photo data.")

    # ==========================================================================
    # TAB 2: EDIT ATTENDANCE (SLIDING TOGGLES & PDF/EXCEL EXPORTS)
    # ==========================================================================
    with tab_att_edit:
        st.subheader("Edit Attendance Roster & Manual Toggle Overrides")
        st.markdown("Captured student face detections default to **Present**. Slide the toggle switch next to any student to adjust their status between Present and Absent.")

        all_db_students = database.get_all_students()
        if not all_db_students:
            st.warning("No registered students found in database.")
        else:
            existing_records = st.session_state.get("last_attendance_records", [])
            existing_map = {r["Register No"]: r for r in existing_records}
            
            records = []
            for s in all_db_students:
                reg = s["register_no"]
                if reg in existing_map:
                    records.append(existing_map[reg])
                else:
                    records.append({
                        "Register No": reg,
                        "Name": s["name"],
                        "Department": s["department"],
                        "Status": "Absent"
                    })
            st.session_state["last_attendance_records"] = records
            
            present_records = [r for r in records if r["Status"] == "Present"]
            absent_records = [r for r in records if r["Status"] == "Absent"]
            
            m1, m2, m3 = st.columns(3)
            m1.metric("Total Class Strength", len(records))
            m2.metric("Currently Present (Toggle ON)", len(present_records), delta=f"{(len(present_records)/max(len(records),1))*100:.1f}%")
            m3.metric("Currently Absent (Toggle OFF)", len(absent_records))

            st.markdown("---")
            tb_col1, tb_col2, tb_col3 = st.columns([1, 1, 2])
            with tb_col1:
                if st.button("Mark All as Present", use_container_width=True, key="btn_mark_all_p"):
                    for r in records:
                        r["Status"] = "Present"
                        st.session_state[f"slide_toggle_{r['Register No']}"] = True
                    st.session_state["last_attendance_records"] = records
                    st.rerun()
            with tb_col2:
                if st.button("Mark All as Absent", use_container_width=True, key="btn_mark_all_a"):
                    for r in records:
                        r["Status"] = "Absent"
                        st.session_state[f"slide_toggle_{r['Register No']}"] = False
                    st.session_state["last_attendance_records"] = records
                    st.rerun()
            with tb_col3:
                search_q = st.text_input("Search Student:", placeholder="Filter by Name or Reg No...", key="att_edit_search_input")

            st.markdown("<hr style='margin: 10px 0; border-color: #334155;'/>", unsafe_allow_html=True)

            hdr_c1, hdr_c2, hdr_c3, hdr_c4, hdr_c5 = st.columns([0.6, 1.5, 2, 2, 1.5])
            with hdr_c1: st.markdown("**S.No**")
            with hdr_c2: st.markdown("**Register No**")
            with hdr_c3: st.markdown("**Student Name**")
            with hdr_c4: st.markdown("**Department / Branch**")
            with hdr_c5: st.markdown("**Attendance Status (Slide Toggle)**")
            st.markdown("<hr style='margin: 4px 0 12px 0; border-color: #334155;'/>", unsafe_allow_html=True)

            filtered_recs = records
            if search_q.strip():
                st_term = search_q.strip().lower()
                filtered_recs = [r for r in records if (st_term in r["Register No"].lower() or st_term in r["Name"].lower())]

            status_changed = False
            for idx, r in enumerate(filtered_recs):
                reg = r["Register No"]
                name = r["Name"]
                dept = r["Department"]
                curr_status = r["Status"]
                is_present = (curr_status == "Present")

                c1, c2, c3, c4, c5 = st.columns([0.6, 1.5, 2, 2, 1.5])
                with c1: st.write(f"**{idx + 1}**")
                with c2: st.code(reg, language="text")
                with c3: st.write(f"**{name}**")
                with c4: st.caption(dept)
                with c5:
                    toggle_val = st.toggle("Present" if is_present else "Absent", value=is_present, key=f"slide_toggle_{reg}")
                    if toggle_val != is_present:
                        r["Status"] = "Present" if toggle_val else "Absent"
                        status_changed = True

            if status_changed:
                st.session_state["last_attendance_records"] = records
                st.rerun()

            st.markdown("---")
            st.subheader("Export Official Attendance Reports")
            st.markdown("Official reports include verified **GPS Location**, **YOLO Detection Engine**, and Present Students roster.")

            cur_slot = st.session_state.get("last_slot_name", "Slot A")
            cur_date = st.session_state.get("last_process_date", datetime.datetime.now().strftime("%Y-%m-%d"))
            cur_time = st.session_state.get("last_process_time", datetime.datetime.now().strftime("%H:%M:%S"))
            
            p_cnt = len(present_records)
            a_cnt = len(absent_records)
            tot_cnt = len(records)

            last_m = st.session_state.get("last_metrics", {})
            loc_str = "Not Permitted / Unavailable"
            if last_m.get("location_lat") is not None and last_m.get("location_lon") is not None:
                loc_str = f"{last_m['location_lat']:.5f}, {last_m['location_lon']:.5f}"
                if last_m.get("location_address"):
                    loc_str += f" ({last_m['location_address']})"
            
            yolo_mod = last_m.get("yolo_model", "yolov8n.pt")

            excel_bytes = export_utils.create_excel_report(
                slot_name=cur_slot,
                faculty_name=faculty_name,
                date_str=cur_date,
                time_str=cur_time,
                total_strength=tot_cnt,
                present_count=p_cnt,
                absent_count=a_cnt,
                df_records=pd.DataFrame(records),
                location_str=loc_str,
                yolo_model_name=yolo_mod
            )

            pdf_bytes = export_utils.create_pdf_report(
                slot_name=cur_slot,
                faculty_name=faculty_name,
                date_str=cur_date,
                time_str=cur_time,
                total_strength=tot_cnt,
                present_count=p_cnt,
                absent_count=a_cnt,
                df_records=pd.DataFrame(records),
                annotated_images=st.session_state.get("last_annotated_imgs", []),
                location_str=loc_str,
                yolo_model_name=yolo_mod
            )

            date_file_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            exp_col1, exp_col2 = st.columns(2)
            with exp_col1:
                st.download_button(
                    label="Download Official Attendance PDF (.pdf)",
                    data=pdf_bytes,
                    file_name=f"Attendance_Report_{date_file_str}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                    key="btn_down_pdf_edit"
                )
            with exp_col2:
                st.download_button(
                    label="Download Attendance Excel Sheet (.xlsx)",
                    data=excel_bytes,
                    file_name=f"Attendance_Report_{date_file_str}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                    key="btn_down_excel_edit"
                )

    # ==========================================================================
    # TAB 3: ALL-TIME ATTENDANCE HISTORY (INTERACTIVE CALENDAR VIEW)
    # ==========================================================================
    with tab_att_history:
        st.subheader("Attendance History & Calendar View")
        st.markdown(f"Displaying attendance records captured by **{faculty_name}**.")

        saved_sessions = database.get_all_attendance_sessions(faculty_name=faculty_name)

        if not saved_sessions:
            st.info(f"No attendance sessions recorded for {faculty_name} yet.")
        else:
            sessions_by_date = {}
            for s in saved_sessions:
                d = s["date_str"]
                if d not in sessions_by_date:
                    sessions_by_date[d] = []
                sessions_by_date[d].append(s)

            now_dt = datetime.datetime.now()
            current_yr = now_dt.year
            current_mo = now_dt.month

            db_years = sorted(list(set([int(s["date_str"].split("-")[0]) for s in saved_sessions if "-" in s["date_str"]] + [current_yr])), reverse=True)
            month_names = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]

            col_y, col_m, col_summary = st.columns([1.5, 1.5, 2])
            with col_y:
                sel_year = st.selectbox("Select Year:", db_years, index=0, key="cal_sel_year")
            with col_m:
                default_m_idx = current_mo - 1
                sel_month_name = st.selectbox("Select Month:", month_names, index=default_m_idx, key="cal_sel_month")
                sel_month_num = month_names.index(sel_month_name) + 1
            with col_summary:
                st.metric("Total Saved Sessions", len(saved_sessions))

            st.markdown("---")
            st.markdown(f"### Interactive Calendar: {sel_month_name} {sel_year}")
            st.markdown("Click any active date button to open attendance records for that day.")

            days_header = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
            hdr_cols = st.columns(7)
            for idx, day_name in enumerate(days_header):
                with hdr_cols[idx]:
                    st.markdown(f"<div style='text-align:center; font-weight:700; color:#38bdf8; font-size:0.85rem;'>{day_name}</div>", unsafe_allow_html=True)

            month_matrix = calendar.monthcalendar(sel_year, sel_month_num)

            for week in month_matrix:
                week_cols = st.columns(7)
                for day_idx, day_num in enumerate(week):
                    with week_cols[day_idx]:
                        if day_num == 0:
                            st.markdown("<div style='height:44px;'></div>", unsafe_allow_html=True)
                        else:
                            date_fmt = f"{sel_year:04d}-{sel_month_num:02d}-{day_num:02d}"
                            has_recs = date_fmt in sessions_by_date
                            
                            if has_recs:
                                sess_cnt = len(sessions_by_date[date_fmt])
                                st.markdown(f"<div style='background:#15803d; border:2px solid #22c55e; border-radius:8px; text-align:center; padding:8px 4px; color:#fff; font-weight:700; font-size:0.95rem; margin-bottom:2px;'>{day_num}<br><span style='font-size:0.65rem; opacity:0.9;'>{sess_cnt} rec</span></div>", unsafe_allow_html=True)
                                if st.button("View", key=f"cal_btn_{date_fmt}", use_container_width=True):
                                    st.session_state["selected_cal_date"] = date_fmt
                                    st.rerun()
                            else:
                                st.markdown(f"<div style='background:#7f1d1d; border:2px solid #ef4444; border-radius:8px; text-align:center; padding:8px 4px; color:#fca5a5; font-weight:600; font-size:0.95rem; margin-bottom:2px; opacity:0.7;'>{day_num}</div>", unsafe_allow_html=True)

            st.markdown("---")
            active_date = st.session_state.get("selected_cal_date", None)

            if active_date:
                st.markdown(f"### Attendance Records for Date: **{active_date}**")
                date_sessions = sessions_by_date.get(active_date, [])

                if not date_sessions:
                    st.info(f"No records stored for date {active_date}.")
                else:
                    for s in date_sessions:
                        s_id = s["id"]
                        s_title = f"Slot: {s['slot_name']} (Time: {s['time_str']} | Faculty: {s['faculty_name']}) — Present: {s['present_count']}/{s['total_students']}"
                        
                        with st.expander(s_title, expanded=True):
                            info, logs = database.get_attendance_session_details(s_id)
                            if info:
                                inf_c1, inf_c2, inf_c3, inf_c4 = st.columns(4)
                                inf_c1.metric("Date & Time", f"{info['date_str']} {info['time_str']}")
                                inf_c2.metric("Total Strength", info['total_students'])
                                inf_c3.metric("Present Count", info['present_count'])
                                inf_c4.metric("Absent Count", info['absent_count'])

                                # Display GPS and Model info if saved
                                loc_lat = info.get("location_lat")
                                loc_lon = info.get("location_lon")
                                if loc_lat is not None and loc_lon is not None:
                                    addr_info = info.get("location_address") or "GPS fix recorded"
                                    st.info(f"📍 **Verified GPS Coordinates:** `{loc_lat:.5f}° N, {loc_lon:.5f}° E` — {addr_info}")
                                else:
                                    st.caption("📍 GPS Location: Not Recorded (Access Denied or Offline)")

                                y_mod = info.get("yolo_model_name")
                                if y_mod:
                                    det_c = info.get("detection_count", 0)
                                    st.caption(f"🤖 **Detection Model:** {y_mod} | Detections: {det_c}")

                                df_logs = pd.DataFrame(logs)
                                st.markdown("#### Present Students Roster")
                                df_logs_present = df_logs[df_logs["Status"] == "Present"].reset_index(drop=True)
                                if not df_logs_present.empty:
                                    df_logs_present.insert(0, "S.No", range(1, len(df_logs_present) + 1))
                                    st.dataframe(df_logs_present, use_container_width=True, hide_index=True)
                                else:
                                    st.info("No present students recorded for this session.")

                                hist_loc_str = f"{loc_lat:.5f}, {loc_lon:.5f}" if (loc_lat and loc_lon) else "Not Recorded"
                                h_pdf_bytes = export_utils.create_pdf_report(
                                    slot_name=info["slot_name"],
                                    faculty_name=info["faculty_name"],
                                    date_str=info["date_str"],
                                    time_str=info["time_str"],
                                    total_strength=info["total_students"],
                                    present_count=info["present_count"],
                                    absent_count=info["absent_count"],
                                    df_records=df_logs,
                                    location_str=hist_loc_str,
                                    yolo_model_name=y_mod
                                )

                                h_excel_bytes = export_utils.create_excel_report(
                                    slot_name=info["slot_name"],
                                    faculty_name=info["faculty_name"],
                                    date_str=info["date_str"],
                                    time_str=info["time_str"],
                                    total_strength=info["total_students"],
                                    present_count=info["present_count"],
                                    absent_count=info["absent_count"],
                                    df_records=df_logs,
                                    location_str=hist_loc_str,
                                    yolo_model_name=y_mod
                                )

                                h_col1, h_col2, h_col3 = st.columns([2, 2, 1])
                                with h_col1:
                                    st.download_button(
                                        label=f"Download PDF Report ({info['date_str']})",
                                        data=h_pdf_bytes,
                                        file_name=f"Attendance_{info['date_str']}_{s_id}.pdf",
                                        mime="application/pdf",
                                        use_container_width=True,
                                        key=f"cal_pdf_{s_id}"
                                    )
                                Promotion_cal_excel_key = f"cal_excel_{s_id}"
                                with h_col2:
                                    st.download_button(
                                        label=f"Download Excel Sheet ({info['date_str']})",
                                        data=h_excel_bytes,
                                        file_name=f"Attendance_{info['date_str']}_{s_id}.xlsx",
                                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                        use_container_width=True,
                                        key=Promotion_cal_excel_key
                                    )
                                with h_col3:
                                    if st.button("Delete Session", key=f"cal_del_{s_id}", use_container_width=True):
                                        database.delete_attendance_session(s_id)
                                        st.session_state["selected_cal_date"] = None
                                        st.success("Session record deleted successfully.")
                                        st.rerun()
            else:
                st.info("Click on any active date in the calendar above to view detailed attendance logs and export PDF/Excel reports.")

    # ==========================================================================
    # TAB 4: MANAGE REGISTERED STUDENTS DIRECTORY
    # ==========================================================================
    with tab_students_mgmt:
        st.subheader("Registered Students Directory & Profile Management")
        all_students = database.get_all_students()
        
        if not all_students:
            st.warning("No students currently registered in database.")
        else:
            sm1, sm2 = st.columns(2)
            sm1.metric("Total Registered Students", len(all_students))
            sm2.metric("Central Database File", "attendance_system.db")
            st.markdown("---")
            
            filtered_students = [{
                "Register No": s["register_no"],
                "Full Name": s["name"],
                "Department / Branch": s["department"]
            } for s in all_students]
            
            st.dataframe(pd.DataFrame(filtered_students), use_container_width=True, hide_index=True)
            
            st.markdown("---")
            st.subheader("Modify or Delete Student Profile")
            student_map = {f"{s['register_no']} - {s['name']} ({s['department']})": s for s in all_students}
            selected_option = st.selectbox("Select a Student to Manage:", list(student_map.keys()), key="mgmt_select_student")
            
            if selected_option:
                target_student = student_map[selected_option]
                target_reg = target_student["register_no"]
                edit_col, delete_col = st.columns([1.2, 1], gap="large")
                
                with edit_col:
                    st.markdown("#### Modify Student Details")
                    with st.form(f"mgmt_edit_form_{target_reg}"):
                        new_name = st.text_input("Full Name", value=target_student["name"])
                        all_dept_options = ["Computer Science & Engineering", "Information Technology", "Electrical & Electronics", "Mechanical Engineering", "Civil Engineering", "Biotechnology", "Other"]
                        curr_dept = target_student["department"]
                        default_idx = all_dept_options.index(curr_dept) if curr_dept in all_dept_options else len(all_dept_options) - 1
                        new_dept = st.selectbox("Department", all_dept_options, index=default_idx)
                        save_btn = st.form_submit_button("Save Updated Details", use_container_width=True)
                        
                        if save_btn:
                            if not new_name.strip():
                                st.error("Full name cannot be empty.")
                            else:
                                success = database.update_student(target_reg, new_name, new_dept)
                                if success:
                                    st.success(f"Details updated for student {target_reg}.")
                                    st.rerun()
                                else:
                                    st.error("Failed to update student details.")

                with delete_col:
                    st.markdown("#### Delete Student Profile")
                    st.warning(f"Are you sure you want to delete {target_student['name']} ({target_reg})?")
                    if st.button(f"Delete Student {target_reg}", use_container_width=True, key=f"mgmt_del_btn_{target_reg}"):
                        success = database.delete_student(target_reg)
                        if success:
                            st.success(f"Student {target_reg} removed from database.")
                            st.rerun()
                        else:
                            st.error("Failed to delete student.")

    # ==========================================================================
    # TAB 5: YOLO DEEP-LEARNING STUDIO & MODEL TRAINING
    # ==========================================================================
    with tab_yolo_studio:
        st.subheader("YOLO Deep-Learning Model Studio & Metrics Analytics")
        st.markdown("Monitor model architecture, inspect genuine loss and evaluation metrics, and run training workflows.")

        # Section 1: Active Model Specification
        st.markdown("### 1. Active Deep-Learning Model Specifications")
        model_info = yolo_workflow.get_yolo_model_info("yolov8n.pt")
        
        sp_c1, sp_c2, sp_c3, sp_c4 = st.columns(4)
        with sp_c1:
            st.metric("Model Architecture", "YOLOv8 Nano (Ultralytics)")
        with sp_c2:
            st.metric("Active Weights", model_info.get("model_name", "yolov8n.pt"))
        with sp_c3:
            st.metric("Parameters", model_info.get("parameters", "3.2M"))
        with sp_c4:
            st.metric("Compute Device", model_info.get("device", "CPU").upper())

        # Section 2: Deep-Learning Metrics Dashboard
        st.markdown("---")
        st.markdown("### 2. Deep-Learning Training & Validation Metrics")
        
        latest_metrics = yolo_workflow.get_latest_training_metrics()
        status = latest_metrics.get("status", "UNTRAINED")

        if status == "UNTRAINED":
            st.warning(
                "⚠️ **Status: Baseline Pre-Trained Weights Active (`yolov8n.pt`)**\n\n"
                "No custom fine-tuning runs have been performed yet on a local classroom dataset. "
                "The metrics below display `N/A` because training loss and mAP are computed during training/validation. "
                "To calculate custom metrics for your environment, configure and train a dataset in the training section below."
            )
            
            dm1, dm2, dm3, dm4 = st.columns(4)
            dm1.metric("Current Epoch", "0 / 0", help="No fine-tuning epochs executed yet.")
            dm2.metric("Training Loss", "N/A", help="Total training loss (Box + Cls + DFL).")
            dm3.metric("Precision (B)", "N/A", help="Genuine bounding-box precision.")
            dm4.metric("Recall (B)", "N/A", help="Genuine bounding-box recall.")

            dm5, dm6, dm7, dm8 = st.columns(4)
            dm5.metric("mAP50", "N/A", help="Mean Average Precision at IoU=0.50.")
            dm6.metric("mAP50-95", "N/A", help="Mean Average Precision across IoU 0.50 to 0.95.")
            dm7.metric("Box Loss", "N/A", help="Bounding box regression loss.")
            dm8.metric("Training Progress", "0.0%", help="Overall epoch progress.")
            
        elif status == "COMPLETED":
            st.success(
                f"✅ **Status: Fine-Tuning Run Completed on {latest_metrics.get('completed_at', 'Recently')}**\n\n"
                f"Trained on dataset: `{latest_metrics.get('dataset_name', 'Custom Dataset')}` | "
                f"Duration: `{latest_metrics.get('duration_seconds', 'N/A')}s`"
            )
            
            cur_ep = latest_metrics.get("epochs_trained", 0)
            tot_ep = latest_metrics.get("total_epochs", 0)
            
            dm1, dm2, dm3, dm4 = st.columns(4)
            dm1.metric("Epochs", f"{cur_ep} / {tot_ep}")
            dm2.metric("Total Loss", f"{latest_metrics.get('total_loss', 'N/A')}")
            dm3.metric("Precision (B)", f"{latest_metrics.get('precision', 'N/A')}")
            dm4.metric("Recall (B)", f"{latest_metrics.get('recall', 'N/A')}")

            dm5, dm6, dm7, dm8 = st.columns(4)
            dm5.metric("mAP50 (B)", f"{latest_metrics.get('map50', 'N/A')}")
            dm6.metric("mAP50-95 (B)", f"{latest_metrics.get('map50_95', 'N/A')}")
            dm7.metric("Box Loss", f"{latest_metrics.get('box_loss', 'N/A')}")
            dm8.metric("Training Progress", f"{latest_metrics.get('progress_pct', 100.0)}%")
        else:
            st.info(f"Training status: {status}")

        # Section 3: Genuine YOLO Training & Fine-Tuning Workflow
        st.markdown("---")
        st.markdown("### 3. YOLO Model Training & Fine-Tuning Workflow")
        st.markdown("Fine-tune YOLO on an annotated classroom dataset. Results are generated from authentic PyTorch execution.")

        t_col1, t_col2 = st.columns([1.5, 1])
        with t_col1:
            dataset_yaml_path = st.text_input(
                "Dataset Configuration Path (data.yaml) *",
                value="yolo_dataset/data.yaml",
                key="yolo_data_yaml_path",
                help="Path to the YOLO-format YAML configuration file pointing to train and val image sets."
            )
            
            hyp_col1, hyp_col2, hyp_col3 = st.columns(3)
            with hyp_col1:
                t_epochs = st.slider("Total Epochs:", min_value=1, max_value=50, value=5, step=1, key="yolo_epochs_slider")
            with hyp_col2:
                t_batch = st.select_slider("Batch Size:", options=[2, 4, 8, 16], value=4, key="yolo_batch_slider")
            with hyp_col3:
                t_lr = st.selectbox("Initial Learning Rate (lr0):", [0.01, 0.005, 0.001], index=0, key="yolo_lr_select")

            action_c1, action_c2 = st.columns(2)
            with action_c1:
                start_train_btn = st.button("🚀 Start YOLO Training Run", use_container_width=True, type="primary", key="btn_start_yolo_train")
            with action_c2:
                validate_btn = st.button("📊 Evaluate / Validate Model", use_container_width=True, key="btn_validate_yolo")

            if start_train_btn:
                if not os.path.exists(dataset_yaml_path):
                    st.error(
                        f"❌ **Missing Dataset Configuration:** File `{dataset_yaml_path}` does not exist on disk.\n\n"
                        "To fine-tune YOLO, an authentic dataset in YOLO format is required. "
                        "Create a `data.yaml` pointing to your annotated classroom images (`train` and `val` directories) "
                        "with class labels. The system does not invent artificial training results."
                    )
                else:
                    st.info(f"Initiating YOLOv8 fine-tuning for {t_epochs} epoch(s)...")
                    p_bar = st.progress(0, text="Initializing training...")
                    status_placeholder = st.empty()

                    def live_training_callback(epoch_data):
                        cur_e = epoch_data.get("epochs_trained", 0)
                        tot_e = epoch_data.get("total_epochs", t_epochs)
                        pct = epoch_data.get("progress_pct", 0.0)
                        p_bar.progress(int(min(pct, 100)), text=f"Epoch {cur_e}/{tot_e} in progress ({pct}%)...")
                        status_placeholder.markdown(
                            f"**Epoch:** {cur_e}/{tot_e} | "
                            f"**Box Loss:** `{epoch_data.get('box_loss')}` | "
                            f"**Total Loss:** `{epoch_data.get('total_loss')}` | "
                            f"**mAP50:** `{epoch_data.get('map50')}`"
                        )

                    with st.spinner("Executing PyTorch training routine..."):
                        run_result = yolo_workflow.train_yolo_model(
                            data_yaml_path=dataset_yaml_path,
                            model_name="yolov8n.pt",
                            epochs=t_epochs,
                            batch_size=t_batch,
                            lr0=t_lr,
                            progress_callback=live_training_callback
                        )

                    if run_result.get("success"):
                        p_bar.progress(100, text="Training completed successfully!")
                        st.success("YOLO training run completed! Real-time metrics have been updated in the dashboard above.")
                        # Save to database as well
                        m_res = run_result.get("metrics", {})
                        database.save_model_training_run(
                            model_name="yolov8n.pt",
                            dataset_name=os.path.basename(dataset_yaml_path),
                            epochs_trained=m_res.get("epochs_trained", t_epochs),
                            total_epochs=t_epochs,
                            final_loss=m_res.get("total_loss"),
                            box_loss=m_res.get("box_loss"),
                            cls_loss=m_res.get("cls_loss"),
                            dfl_loss=m_res.get("dfl_loss"),
                            precision_score=m_res.get("precision"),
                            recall_score=m_res.get("recall"),
                            map50=m_res.get("map50"),
                            map50_95=m_res.get("map50_95"),
                            training_status="COMPLETED",
                            started_at=m_res.get("started_at"),
                            completed_at=m_res.get("completed_at"),
                            notes=f"Trained on {dataset_yaml_path} with batch={t_batch}, lr0={t_lr}"
                        )
                        st.rerun()
                    else:
                        st.error(f"Training failed: {run_result.get('error')}")

            if validate_btn:
                if not os.path.exists(dataset_yaml_path):
                    st.error(f"Cannot validate: Dataset configuration `{dataset_yaml_path}` not found.")
                else:
                    with st.spinner("Running YOLO validation on test/val split..."):
                        val_res = yolo_workflow.validate_yolo_model(data_yaml_path=dataset_yaml_path)
                        if val_res.get("success"):
                            st.success("Validation completed!")
                            v1, v2, v3, v4 = st.columns(4)
                            v1.metric("Val Precision", val_res.get("precision"))
                            v2.metric("Val Recall", val_res.get("recall"))
                            v3.metric("Val mAP50", val_res.get("map50"))
                            v4.metric("Val mAP50-95", val_res.get("map50_95"))
                        else:
                            st.error(f"Validation failed: {val_res.get('error')}")

        with t_col2:
            st.markdown("#### 📁 YOLO Dataset Directory Structure")
            st.code("""
# Required layout for data.yaml:
path: ./yolo_dataset
train: images/train
val: images/val

names:
  0: person
  1: student_face
            """, language="yaml")
            st.caption(
                "Each image in `images/` must have a corresponding `.txt` file in `labels/` "
                "with lines in format: `<class_id> <x_center> <y_center> <width> <height>` (normalized 0 to 1)."
            )

        # Section 4: Live YOLO Inference Tester
        st.markdown("---")
        st.markdown("### 4. Interactive Live YOLO Inference Diagnostic")
        st.markdown("Upload any test image to evaluate YOLO object and person detection in real time.")
        
        test_img_file = st.file_uploader("Upload Test Photo for YOLO Inference:", type=["jpg", "jpeg", "png"], key="yolo_test_up")
        if test_img_file:
            t_col_img, t_col_res = st.columns([1.5, 1])
            test_bgr = load_image_as_bgr(test_img_file)
            if test_bgr is not None:
                with st.spinner("Running YOLO detection..."):
                    detections, speeds = yolo_workflow.detect_people_yolo(test_bgr, conf_threshold=0.25)
                    
                    # Annotate image
                    annotated_test = test_bgr.copy()
                    for det in detections:
                        b = det["bbox"]
                        cv2.rectangle(annotated_test, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)
                        cv2.putText(
                            annotated_test,
                            f"{det['class_name']} [{det['confidence']:.2f}]",
                            (b[0], max(15, b[1] - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 0),
                            1
                        )
                    annotated_rgb_test = cv2.cvtColor(annotated_test, cv2.COLOR_BGR2RGB)

                with t_col_img:
                    st.image(annotated_rgb_test, caption="YOLO Bounding Box Localization Result", use_container_width=True)
                with t_col_res:
                    st.markdown("#### Detection Metrics")
                    st.metric("Total Detections", len(detections))
                    st.metric("Inference Latency", f"{speeds['inference_ms']} ms")
                    st.metric("Preprocess Latency", f"{speeds['preprocess_ms']} ms")
                    st.metric("Postprocess Latency", f"{speeds['postprocess_ms']} ms")
                    st.metric("Total Processing Time", f"{speeds['total_ms']} ms")
                    if detections:
                        st.dataframe(pd.DataFrame(detections)[["class_name", "confidence", "bbox"]], use_container_width=True, hide_index=True)
