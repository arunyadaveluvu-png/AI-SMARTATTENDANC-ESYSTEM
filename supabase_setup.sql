-- ============================================================
-- AI Smart Attendance System - Supabase Database Setup
-- Run this in Supabase SQL Editor: https://supabase.com/dashboard
-- ============================================================

-- 1. Students table (stores registration details as JSON-compatible records)
CREATE TABLE IF NOT EXISTS students (
    register_no TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    department  TEXT NOT NULL,
    embedding   TEXT NOT NULL   -- base64-encoded pickle of face embedding
);

-- Enable Row Level Security
ALTER TABLE students ENABLE ROW LEVEL SECURITY;

-- Allow anon key full access (adjust for production)
CREATE POLICY "Allow all on students" ON students
    FOR ALL USING (true) WITH CHECK (true);

-- 2. Faculty table
CREATE TABLE IF NOT EXISTS faculty (
    username TEXT PRIMARY KEY,
    password TEXT NOT NULL,
    name     TEXT NOT NULL
);

ALTER TABLE faculty ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Allow all on faculty" ON faculty
    FOR ALL USING (true) WITH CHECK (true);

-- Seed default faculty admin
INSERT INTO faculty (username, password, name)
VALUES ('faculty', 'password123', 'Faculty Admin')
ON CONFLICT (username) DO NOTHING;

-- 3. Attendance Sessions table
CREATE TABLE IF NOT EXISTS attendance_sessions (
    id               BIGSERIAL PRIMARY KEY,
    slot_name        TEXT    NOT NULL,
    faculty_name     TEXT    NOT NULL,
    date_str         TEXT    NOT NULL,
    time_str         TEXT    NOT NULL,
    total_students   INTEGER NOT NULL,
    present_count    INTEGER NOT NULL,
    absent_count     INTEGER NOT NULL,
    location_lat     REAL,
    location_lon     REAL,
    location_address TEXT,
    yolo_model_name  TEXT,
    detection_count  INTEGER,
    created_at       TIMESTAMPTZ DEFAULT NOW()
);

ALTER TABLE attendance_sessions ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Allow all on attendance_sessions" ON attendance_sessions
    FOR ALL USING (true) WITH CHECK (true);

-- 4. Attendance Logs table
CREATE TABLE IF NOT EXISTS attendance_logs (
    id          BIGSERIAL PRIMARY KEY,
    session_id  BIGINT  NOT NULL REFERENCES attendance_sessions(id) ON DELETE CASCADE,
    register_no TEXT    NOT NULL,
    name        TEXT    NOT NULL,
    department  TEXT    NOT NULL,
    status      TEXT    NOT NULL
);

ALTER TABLE attendance_logs ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Allow all on attendance_logs" ON attendance_logs
    FOR ALL USING (true) WITH CHECK (true);

-- 5. Model Training History table
CREATE TABLE IF NOT EXISTS model_training_history (
    id               BIGSERIAL PRIMARY KEY,
    model_name       TEXT NOT NULL,
    dataset_name     TEXT,
    epochs_trained   INTEGER NOT NULL,
    total_epochs     INTEGER NOT NULL,
    final_loss       REAL,
    box_loss         REAL,
    cls_loss         REAL,
    dfl_loss         REAL,
    precision_score  REAL,
    recall_score     REAL,
    map50            REAL,
    map50_95         REAL,
    training_status  TEXT NOT NULL,
    started_at       TEXT,
    completed_at     TEXT,
    notes            TEXT
);

ALTER TABLE model_training_history ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Allow all on model_training_history" ON model_training_history
    FOR ALL USING (true) WITH CHECK (true);

-- 6. System Settings table
CREATE TABLE IF NOT EXISTS system_settings (
    setting_key   TEXT PRIMARY KEY,
    setting_value TEXT NOT NULL
);

ALTER TABLE system_settings ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Allow all on system_settings" ON system_settings
    FOR ALL USING (true) WITH CHECK (true);
