import streamlit as st
import cv2
import numpy as np
import pandas as pd

import database
import importlib
importlib.reload(database)
import face_utils

# Page Config
st.set_page_config(
    page_title="Registration Portal - AI Attendance System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Professional Executive Theme Alignment
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

    /* Headers */
    .main-header {
        font-size: 2.1rem;
        font-weight: 800;
        color: #38bdf8 !important;
        margin-bottom: 0.25rem;
    }
    .sub-header {
        font-size: 0.95rem;
        color: #94a3b8;
        margin-bottom: 1.5rem;
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
    
    /* Buttons Contrast Fix */
    .stButton > button, div[data-testid="stFormSubmitButton"] > button {
        background-color: #0284c7 !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        border: none !important;
        border-radius: 6px !important;
        transition: all 0.2s ease-in-out;
    }
    .stButton > button:hover, div[data-testid="stFormSubmitButton"] > button:hover {
        background-color: #0369a1 !important;
        color: #ffffff !important;
        box-shadow: 0 4px 12px rgba(2, 132, 199, 0.3);
    }
    
    div[data-baseweb="input"] {
        background-color: #1e293b !important;
        border-color: #334155 !important;
        border-radius: 6px !important;
    }
    div[data-baseweb="input"] input {
        color: #f8fafc !important;
    }
    div[data-baseweb="input"] button {
        background-color: transparent !important;
        color: #94a3b8 !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Database
database.init_db()

# Sidebar Info
st.sidebar.title("Registration System")
st.sidebar.info(
    "Local JSON Storage\n\n"
    "All registered profiles are saved to .local/ JSON files."
)

st.markdown('<div class="main-header">Student & Faculty Registration</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Register student facial profiles and faculty accounts into the database.</div>', unsafe_allow_html=True)

# Tabs Navigation
tab1, tab2 = st.tabs(["Student Face Registration", "Faculty Account Registration"])

def load_image_as_bgr(uploaded_file) -> np.ndarray:
    bytes_data = uploaded_file.getvalue()
    file_bytes = np.asarray(bytearray(bytes_data), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return img_bgr

# ==============================================================================
# TAB 1: STUDENT REGISTRATION
# ==============================================================================
with tab1:
    col1, col2 = st.columns([1.2, 1], gap="large")

    with col1:
        st.subheader("Student Details & Face Upload")
        with st.form("student_reg_form", clear_on_submit=False):
            reg_no = st.text_input("Register Number *", placeholder="e.g. 24BCA8011")
            full_name = st.text_input("Full Name *", placeholder="e.g. Jane Doe")
            department = st.selectbox(
                "Department *",
                ["Computer Science & Engineering", "Information Technology", "Electrical & Electronics", "Mechanical Engineering", "Civil Engineering", "Biotechnology", "Other"]
            )
            
            st.markdown("### Face Photo Capture")
            input_source = st.radio("Choose Photo Input Method:", ["Upload Photo", "Live Webcam Snapshot"], horizontal=True)
            
            photo_file = None
            if input_source == "Upload Photo":
                photo_file = st.file_uploader("Upload Clear Face Photo (JPG, PNG)", type=["jpg", "jpeg", "png"])
            else:
                photo_file = st.camera_input("Take a Snapshot")
            
            submit_btn = st.form_submit_button("Register Student Profile", use_container_width=True)

        if submit_btn:
            if not reg_no.strip() or not full_name.strip():
                st.error("Please fill in all required fields (Register Number and Full Name).")
            elif photo_file is None:
                st.error("Please upload or capture a photo.")
            elif database.student_exists(reg_no):
                st.error(f"Student with Register Number '{reg_no.strip()}' is already registered in database!")
            else:
                with st.spinner("Processing facial embedding..."):
                    img_bgr = load_image_as_bgr(photo_file)
                    
                    if img_bgr is None:
                        st.error("Failed to decode image. Please upload a valid image file.")
                    else:
                        embedding, status_code, message = face_utils.extract_single_face_embedding(img_bgr)
                        
                        if status_code == "NO_FACE":
                            st.error(f"{message}")
                        elif status_code == "MULTIPLE_FACES":
                            st.warning(f"{message}")
                        elif status_code == "SUCCESS":
                            success = database.add_student(reg_no, full_name, department, embedding)
                            if success:
                                st.success(f"Success! Student '{full_name}' ({reg_no}) registered successfully.")
                            else:
                                st.error("Database error during insertion.")
                        else:
                            st.error(f"{message}")

    with col2:
        st.subheader("Registration Database Status")
        st.success("Cloud Database Connected (Supabase)")
            
        all_registered = database.get_all_students()
        st.metric("Total Registered Students", len(all_registered))
        
        if all_registered:
            with st.expander("Registered Students Roster", expanded=False):
                df_students = pd.DataFrame([
                    {"Register No": s["register_no"], "Name": s["name"], "Department": s["department"]}
                    for s in all_registered
                ])
                st.dataframe(df_students, use_container_width=True, hide_index=True)

# ==============================================================================
# TAB 2: FACULTY REGISTRATION
# ==============================================================================
with tab2:
    f_col1, f_col2 = st.columns([1.2, 1], gap="large")
    
    with f_col1:
        st.subheader("New Faculty Account Registration")
        with st.form("faculty_reg_form", clear_on_submit=True):
            f_name = st.text_input("Full Name *", placeholder="Dr. Alan Turing")
            f_username = st.text_input("Username *", placeholder="alan_turing")
            f_password = st.text_input("Password *", type="password", placeholder="Set secure password")
            f_confirm_pw = st.text_input("Confirm Password *", type="password")
            
            f_submit = st.form_submit_button("Register Faculty Account", use_container_width=True)
            
        if f_submit:
            if not f_name.strip() or not f_username.strip() or not f_password.strip():
                st.error("All fields are required.")
            elif f_password != f_confirm_pw:
                st.error("Passwords do not match!")
            elif database.faculty_exists(f_username):
                st.error(f"Username '{f_username.strip()}' is already taken.")
            else:
                success = database.add_faculty(f_username, f_password, f_name)
                if success:
                    st.success(f"Faculty account created for '{f_name}' (Username: {f_username.strip()}). You can now log in.")
                else:
                    st.error("Database error while creating faculty account.")
                    
    with f_col2:
        st.subheader("Registered Faculty Accounts")
        faculty_list = database.get_all_faculty()
        st.metric("Total Registered Faculty", len(faculty_list))
        
        if faculty_list:
            df_fac = pd.DataFrame([
                {"Username": f["username"], "Name": f["name"]}
                for f in faculty_list
            ])
            st.dataframe(df_fac, use_container_width=True, hide_index=True)
