import streamlit as st
import cv2
import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import database
import face_utils
import importlib
importlib.reload(database)

st.set_page_config(
    page_title="Registration Portal – AI Attendance",
    page_icon="👤",
    layout="wide",
    initial_sidebar_state="expanded"
)

database.init_db()

# ── Styling ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
html, body, .stApp { background-color: #0f172a !important; color: #f8fafc !important;
    font-family: 'Inter', sans-serif !important; }
[data-testid="stSidebar"] { background-color: #1e293b !important; border-right: 1px solid #334155 !important; }
.stTabs [data-baseweb="tab-list"] { background-color: #1e293b !important; padding: 6px;
    border-radius: 8px; border: 1px solid #334155 !important; }
.stTabs [data-baseweb="tab"] { color: #94a3b8 !important; font-weight: 600; }
.stTabs [aria-selected="true"] { background-color: #0284c7 !important; color: #fff !important; border-radius: 6px; }
.stButton > button { background-color: #0284c7 !important; color: #fff !important;
    font-weight: 600 !important; border-radius: 6px !important; border: none !important; }
div[data-baseweb="input"] { background-color: #1e293b !important; border-color: #334155 !important; }
div[data-baseweb="input"] input { color: #f8fafc !important; }
</style>
""", unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style="background:linear-gradient(135deg,#0284c7,#0369a1 60%,#0f172a);
    padding:1.5rem 2rem;border-radius:12px;border-top:4px solid #38bdf8;margin-bottom:1.5rem;">
  <div style="font-size:1.9rem;font-weight:800;color:#fff;">👤 Student & Faculty Registration</div>
  <div style="font-size:0.9rem;color:#38bdf8;font-weight:600;margin-top:4px;">
    Register student face profiles and faculty accounts · Local JSON Storage
  </div>
</div>
""", unsafe_allow_html=True)

# ── Webcam Notice ─────────────────────────────────────────────────────────────
st.info(
    "📷 **Live Webcam Tip:** The camera widget works over **HTTPS** (Streamlit Cloud is HTTPS ✅). "
    "When prompted, click **Allow** in your browser to grant camera access. "
    "If the camera is blocked, switch to **Upload Photo** instead."
)

# ── Helpers ───────────────────────────────────────────────────────────────────
def load_image_as_bgr(uploaded_file) -> np.ndarray:
    file_bytes = np.asarray(bytearray(uploaded_file.getvalue()), dtype=np.uint8)
    return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🧑‍🎓 Student Face Registration", "👩‍🏫 Faculty Account Registration"])

# ==============================================================================
# TAB 1: STUDENT REGISTRATION
# ==============================================================================
with tab1:
    col1, col2 = st.columns([1.2, 1], gap="large")

    with col1:
        st.subheader("Student Details & Face Upload")
        with st.form("reg_student_form", clear_on_submit=False):
            reg_no    = st.text_input("Register Number *", placeholder="e.g. 24BCA8011")
            full_name = st.text_input("Full Name *", placeholder="e.g. Jane Doe")
            department = st.selectbox("Department *", [
                "Computer Science & Engineering", "Information Technology",
                "Electrical & Electronics", "Mechanical Engineering",
                "Civil Engineering", "Biotechnology", "Other"
            ])

            st.markdown("#### Face Photo")
            input_src = st.radio(
                "Choose input method:",
                ["📁 Upload Photo", "📷 Live Webcam Snapshot"],
                horizontal=True,
                key="reg_input_src"
            )

            photo_file = None
            if input_src == "📁 Upload Photo":
                photo_file = st.file_uploader(
                    "Upload a clear face photo (JPG / PNG)",
                    type=["jpg", "jpeg", "png"],
                    key="reg_file_up"
                )
            else:
                st.caption("⚠️ Allow camera access in your browser when prompted.")
                photo_file = st.camera_input("📷 Take a Snapshot", key="reg_cam_snap")

            submit_btn = st.form_submit_button("Register Student Profile", use_container_width=True)

        if submit_btn:
            if not reg_no.strip() or not full_name.strip():
                st.error("Please fill in Register Number and Full Name.")
            elif photo_file is None:
                st.error("Please upload or capture a photo.")
            elif database.student_exists(reg_no):
                st.error(f"Register Number '{reg_no.strip()}' is already registered!")
            else:
                with st.spinner("Processing facial embedding…"):
                    img_bgr = load_image_as_bgr(photo_file)
                    if img_bgr is None:
                        st.error("Failed to decode image. Please use a valid JPG or PNG.")
                    else:
                        embedding, status_code, message = face_utils.extract_single_face_embedding(img_bgr)
                        if status_code == "NO_FACE":
                            st.error(f"❌ {message}")
                        elif status_code == "MULTIPLE_FACES":
                            st.warning(f"⚠️ {message}")
                        elif status_code == "SUCCESS":
                            if database.add_student(reg_no, full_name, department, embedding):
                                st.success(f"✅ Student '{full_name}' ({reg_no.strip()}) registered successfully!")
                            else:
                                st.error("Database error — could not insert student.")
                        else:
                            st.error(f"{message}")

    with col2:
        st.subheader("Registration Status")
        st.success("✅ Local JSON Storage Active")
        all_students = database.get_all_students()
        st.metric("Total Registered Students", len(all_students))

        st.markdown("#### Registration Tips")
        st.markdown("""
- **Single face** — only you should be visible  
- **Good lighting** — bright, even light on your face  
- **Look straight** — face the camera directly  
- **No obstruction** — remove sunglasses / heavy masks
        """)

        if all_students:
            with st.expander("Registered Students Roster", expanded=False):
                df = pd.DataFrame([
                    {"Register No": s["register_no"], "Name": s["name"], "Department": s["department"]}
                    for s in all_students
                ])
                st.dataframe(df, use_container_width=True, hide_index=True)

# ==============================================================================
# TAB 2: FACULTY REGISTRATION
# ==============================================================================
with tab2:
    f_col1, f_col2 = st.columns([1.2, 1], gap="large")

    with f_col1:
        st.subheader("New Faculty Account")
        with st.form("reg_faculty_form", clear_on_submit=True):
            f_name       = st.text_input("Full Name *", placeholder="Dr. Alan Turing")
            f_username   = st.text_input("Username *", placeholder="alan_turing")
            f_password   = st.text_input("Password *", type="password")
            f_confirm_pw = st.text_input("Confirm Password *", type="password")
            f_submit     = st.form_submit_button("Register Faculty Account", use_container_width=True)

        if f_submit:
            if not f_name.strip() or not f_username.strip() or not f_password.strip():
                st.error("All fields are required.")
            elif f_password != f_confirm_pw:
                st.error("Passwords do not match!")
            elif database.faculty_exists(f_username):
                st.error(f"Username '{f_username.strip()}' is already taken.")
            else:
                if database.add_faculty(f_username, f_password, f_name):
                    st.success(f"✅ Faculty account created for '{f_name}'. You can now log in.")
                else:
                    st.error("Database error — could not create account.")

    with f_col2:
        st.subheader("Registered Faculty")
        faculty_list = database.get_all_faculty()
        st.metric("Total Faculty Accounts", len(faculty_list))
        if faculty_list:
            st.dataframe(
                pd.DataFrame([{"Username": f["username"], "Name": f["name"]} for f in faculty_list]),
                use_container_width=True, hide_index=True
            )
