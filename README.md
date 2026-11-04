# AI Smart Attendance System 🎓

An AI-powered attendance management system using **face recognition** (InsightFace + YOLOv8) with **local JSON database storage** and a **Streamlit** web UI.

---

## 🌟 Features

- 👤 **Student & Faculty Registration** — enroll face profiles using **Live Webcam** or photo upload
- 🤖 **AI Face Recognition** — InsightFace deep learning facial embeddings
- 📸 **YOLOv8 Crowd Detection** — real-time classroom person counting & localization
- 💾 **Pure JSON Storage** — all data saved directly to standard `.json` files (`students.json`, `faculty.json`, etc.)
- 📊 **Attendance Reports** — download official PDF and Excel reports with one click
- 📍 **GPS Geo-tagging** — automatically stamps verification coordinates on attendance photos
- 🌙 **Executive Dark UI** — clean, responsive design

---

## 🚀 How to Run & Deploy

### Option 1: Unified Application (`app.py`) — Recommended
Contains **both Registration and Attendance** in a single deployed link with clean sidebar portal switching:
```bash
streamlit run app.py
```

### Option 2: Individual Standalone Apps
If you want to deploy registration and attendance on **separate Streamlit links**:
- **Registration Only:** `streamlit run registration_app.py`
- **Attendance Only:** `streamlit run attendance_app.py`

---

## 📷 Webcam Troubleshooting on Streamlit Cloud

When deploying on Streamlit Cloud (`share.streamlit.io`):
1. **Camera Permission:** When you first load the webcam widget, your browser will ask **"Allow site to use your camera?"** — click **Allow**.
2. **HTTPS Required:** Camera access (`getUserMedia`) is restricted by modern browsers to secure HTTPS connections. Streamlit Cloud is HTTPS by default.
3. **Fallback:** If camera access is blocked by your browser or device, switch to the **"Upload Photo"** option to upload a clear face image.

---

## 💾 JSON Database Format

No external database server or Supabase setup is required. All records are stored locally as JSON:

| JSON File | Content |
|-----------|---------|
| `students.json` | Register number, full name, department, base64 face embedding |
| `faculty.json` | Faculty usernames, credentials, and names |
| `attendance_sessions.json` | Session metadata, slot name, timestamp, present count, GPS coords |
| `attendance_logs.json` | Detailed per-student attendance status for each session |
| `model_training_history.json` | Training metrics and run logs |

*Default Faculty Login:* `faculty` / `password123`

---

## 🛠 Tech Stack

- **UI Framework:** Streamlit
- **Face Embedding:** InsightFace (ONNX Runtime)
- **Object / Person Detection:** Ultralytics YOLOv8
- **Data Storage:** Python JSON file engine
- **Reporting:** ReportLab (PDF), OpenPyXL (Excel)
