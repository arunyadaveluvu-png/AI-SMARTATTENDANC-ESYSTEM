import streamlit as st
import database

st.set_page_config(
    page_title="AI Smart Attendance System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

database.init_db()

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
html, body, .stApp { background-color: #0f172a !important; color: #f8fafc !important;
    font-family: 'Inter', sans-serif !important; }
[data-testid="stSidebar"] { background-color: #1e293b !important; border-right: 1px solid #334155 !important; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style="background:linear-gradient(135deg,#0284c7,#0369a1 60%,#0f172a);
    padding:2rem 2.5rem;border-radius:14px;border-top:4px solid #38bdf8;margin-bottom:2rem;">
  <div style="font-size:2.4rem;font-weight:800;color:#fff;letter-spacing:-0.02em;">
    🎓 AI Smart Attendance System
  </div>
  <div style="font-size:1rem;color:#38bdf8;font-weight:600;margin-top:6px;letter-spacing:0.06em;">
    INSIGHTFACE · YOLOV8 · REAL-TIME FACE RECOGNITION
  </div>
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
students = database.get_all_students()
sessions = database.get_all_attendance_sessions()

with col1:
    st.metric("📋 Registered Students", len(students))
with col2:
    st.metric("📅 Attendance Sessions", len(sessions))
with col3:
    present_total = sum(s.get("present_count", 0) for s in sessions)
    st.metric("✅ Total Present Records", present_total)

st.markdown("---")
st.markdown("### 🧭 Navigate to a Portal")

nav1, nav2 = st.columns(2)
with nav1:
    st.markdown("""
    <div style="background:#1e293b;border:1px solid #334155;border-top:3px solid #0284c7;
        border-radius:10px;padding:1.5rem;">
        <div style="font-size:1.3rem;font-weight:700;color:#38bdf8;">👤 Registration Portal</div>
        <div style="color:#94a3b8;margin-top:8px;">Register student face profiles and faculty accounts.</div>
    </div>
    """, unsafe_allow_html=True)
    st.page_link("pages/1_Registration.py", label="Open Registration Portal →", use_container_width=True)

with nav2:
    st.markdown("""
    <div style="background:#1e293b;border:1px solid #334155;border-top:3px solid #22c55e;
        border-radius:10px;padding:1.5rem;">
        <div style="font-size:1.3rem;font-weight:700;color:#4ade80;">📸 Attendance Portal</div>
        <div style="color:#94a3b8;margin-top:8px;">Capture, process and manage classroom attendance.</div>
    </div>
    """, unsafe_allow_html=True)
    st.page_link("pages/2_Attendance.py", label="Open Attendance Portal →", use_container_width=True)

st.markdown("---")
st.info("💡 Use the **sidebar navigation** on the left to switch between portals at any time.")
