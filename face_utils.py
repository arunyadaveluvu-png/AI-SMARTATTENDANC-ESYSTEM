import datetime
import sys
import cv2
import numpy as np
import streamlit as st
from typing import List, Dict, Tuple, Optional, Any
import yolo_workflow
import location_utils

# Try loading InsightFace, with fallback indicator if missing
try:
    import insightface
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False

@st.cache_resource
def load_face_analysis_model():
    """
    Loads and caches the InsightFace FaceAnalysis model.
    """
    if not INSIGHTFACE_AVAILABLE:
        return None
    try:
        app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=0, det_size=(640, 640))
        return app
    except Exception as e:
        # Fallback to default or lighter setup if buffalo_l download/init fails
        try:
            app = FaceAnalysis(providers=["CPUExecutionProvider"])
            app.prepare(ctx_id=0, det_size=(640, 640))
            return app
        except Exception as inner_e:
            print(f"Failed to load InsightFace model: {inner_e}")
            return None

def compute_cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Compute cosine similarity between two feature vectors."""
    dot_product = np.dot(emb1, emb2)
    norm_a = np.linalg.norm(emb1)
    norm_b = np.linalg.norm(emb2)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot_product / (norm_a * norm_b))

def extract_single_face_embedding(image_bgr: np.ndarray) -> Tuple[Optional[np.ndarray], str, str]:
    """
    Extract facial embedding from a single student registration image.
    Validates face count in the frame.
    
    Returns:
        (embedding, status_code, message)
    """
    model = load_face_analysis_model()
    if model is None:
        return None, "MODEL_ERROR", "InsightFace feature extraction model is not initialized."
    
    faces = model.get(image_bgr)
    
    if len(faces) == 0:
        return None, "NO_FACE", "No face detected in the image. Please upload a clear photo showing your face."
    elif len(faces) > 1:
        return None, "MULTIPLE_FACES", f"Detected {len(faces)} faces! Please ensure only your face is present in the frame."
    
    # Single face successfully detected
    embedding = faces[0].embedding
    return embedding, "SUCCESS", "Face extracted successfully!"


def bbox_iou(boxA, boxB) -> float:
    """Compute Intersection over Union between two [x1, y1, x2, y2] bounding boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = max(1, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = max(1, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    return float(interArea / float(boxAArea + boxBArea - interArea))


def print_yolo_terminal_accuracy_report(
    model_name: str,
    yolo_detections: List[Dict[str, Any]],
    speed_metrics: Dict[str, float],
    detected_faces_count: int,
    present_count: int,
    total_students: int,
    avg_face_confidence: Optional[float],
    location_data: Optional[Dict[str, Any]] = None,
    slot_name: str = ""
):
    """
    Prints a prominent, professional YOLO Deep Learning inference, accuracy,
    and person localization report directly to terminal stdout with flush=True.
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    num_yolo = len(yolo_detections)
    confs = [d["confidence"] for d in yolo_detections]
    avg_yolo_conf = (sum(confs) / len(confs)) * 100.0 if confs else 0.0
    match_pct = (present_count / max(total_students, 1)) * 100.0
    total_ms = speed_metrics.get("total_ms", 0.0)
    fps_est = round(1000.0 / total_ms, 1) if total_ms > 0 else 0.0
    
    loc_summary = "GPS Access Disabled / Unavailable"
    if location_data and location_data.get("status") == "GRANTED":
        lat = location_data.get("latitude")
        lon = location_data.get("longitude")
        acc = location_data.get("accuracy", 0.0)
        addr = location_data.get("address") or "Coordinates Verified"
        loc_summary = f"{lat:.6f}° N, {lon:.6f}° E (±{acc:.1f}m) | {addr}"

    banner = [
        "",
        "=" * 84,
        "          🎯 ULTRALYTICS YOLO DEEP-LEARNING INFERENCE & ACCURACY REPORT 🎯          ",
        "=" * 84,
        f"  [SESSION]       : {slot_name or 'Classroom Attendance Photo'}",
        f"  [TIMESTAMP]     : {timestamp}",
        f"  [DEFAULT MODEL] : YOLOv8 Nano ({model_name}) - Ultralytics Deep-Learning Engine",
        "-" * 84,
        "  🎯 YOLOv8 PERSON & OBJECT DETECTION ACCURACY:",
        f"    • Total Persons Localized by YOLOv8 : {num_yolo}",
        f"    • Average YOLOv8 Detection Accuracy : {avg_yolo_conf:.2f}%",
    ]
    
    if confs:
        banner.append("    • Detailed Detection Confidences:")
        for idx, det in enumerate(yolo_detections):
            b = det["bbox"]
            conf_val = det["confidence"] * 100.0
            banner.append(
                f"        - Target #{idx + 1:02d}: Class='{det.get('class_name', 'person')}' | "
                f"Confidence={conf_val:.2f}% | Box=[x1={b[0]}, y1={b[1]}, x2={b[2]}, y2={b[3]}]"
            )
    else:
        banner.append("    • (No persons localized by YOLO in current frame)")

    banner.extend([
        "-" * 84,
        "  🧠 FACIAL EMBEDDING & RECOGNITION MATCH ACCURACY:",
        f"    • InsightFace Detected Faces       : {detected_faces_count}",
        f"    • Enrolled Students Identified     : {present_count} / {total_students}",
        f"    • Attendance Recognition Accuracy  : {match_pct:.1f}%",
        f"    • Avg Facial Cosine Similarity     : {f'{avg_face_confidence:.3f}' if avg_face_confidence else 'N/A'}",
        "-" * 84,
        "  ⚡ LATENCY & INFERENCE PERFORMANCE:",
        f"    • Pre-process Speed  : {speed_metrics.get('preprocess_ms', 0.0)} ms",
        f"    • YOLO Inference     : {speed_metrics.get('inference_ms', 0.0)} ms",
        f"    • Post-process Speed : {speed_metrics.get('postprocess_ms', 0.0)} ms",
        f"    • Total Frame Time   : {total_ms} ms (~{fps_est} FPS)",
        "-" * 84,
        "  📍 GEOLOCATION VERIFICATION:",
        f"    • Watermark Status   : {loc_summary}",
        "=" * 84,
        ""
    ])
    
    sys.stdout.write("\n".join(banner) + "\n")
    sys.stdout.flush()


def recognize_faces_in_group(
    group_image_bgr: np.ndarray,
    registered_students: List[Dict[str, Any]],
    threshold: float = 0.5,
    yolo_model_name: str = "yolov8n.pt",
    yolo_conf_threshold: float = 0.25,
    use_yolo: bool = True,
    location_data: Optional[Dict[str, Any]] = None,
    slot_name: str = "",
    timestamp_str: Optional[str] = None
) -> Tuple[np.ndarray, List[Dict[str, Any]], Dict[str, Any]]:
    """
    Detects faces/people in classroom photo using YOLO Deep Learning detector
    and matches facial embeddings against registered students.
    Watermarks the output image with verified GPS location and prints detailed
    accuracy metrics directly to the system terminal.
    
    Returns:
        (annotated_image_bgr, attendance_records, summary_metrics)
    """
    annotated_img = group_image_bgr.copy()
    face_model = load_face_analysis_model()
    
    # 1. Run YOLO Deep Learning Inference
    yolo_detections = []
    yolo_speed = {"preprocess_ms": 0.0, "inference_ms": 0.0, "postprocess_ms": 0.0, "total_ms": 0.0}
    if use_yolo:
        yolo_detections, yolo_speed = yolo_workflow.detect_people_yolo(
            image_bgr=group_image_bgr,
            conf_threshold=yolo_conf_threshold,
            model_name=yolo_model_name
        )

    # 2. Extract faces via InsightFace
    detected_faces = []
    if face_model is not None:
        try:
            detected_faces = face_model.get(group_image_bgr)
        except Exception as e:
            print(f"FaceAnalysis get exception: {e}")
            detected_faces = []
            
    # Track presence for registered students
    present_reg_nos = set()
    matches_info = {} # reg_no -> best_sim
    present_confidences = []

    # Map each face to best matching YOLO person bounding box if available
    matched_yolo_indices = set()
    
    for face in detected_faces:
        f_bbox = face.bbox.astype(int)
        det_embedding = face.embedding
        
        # Check if this face falls inside any YOLO detected person box
        best_yolo_idx = None
        best_yolo_conf = None
        for idx, y_det in enumerate(yolo_detections):
            y_box = y_det["bbox"]
            # Check overlap or containment
            if (f_bbox[0] >= y_box[0] - 20 and f_bbox[1] >= y_box[1] - 20 and
                f_bbox[2] <= y_box[2] + 20 and f_bbox[3] <= y_box[3] + 20) or bbox_iou(f_bbox, y_box) > 0.15:
                best_yolo_idx = idx
                best_yolo_conf = y_det["confidence"]
                matched_yolo_indices.add(idx)
                break

        best_match_reg = None
        best_sim = -1.0
        best_student = None
        
        # Compare against all registered students
        for student in registered_students:
            sim = compute_cosine_similarity(det_embedding, student["embedding"])
            if sim > best_sim:
                best_sim = sim
                best_match_reg = student["register_no"]
                best_student = student
        
        # Determine annotation bounding box: face bbox
        box_to_draw = f_bbox
        
        if best_sim >= threshold and best_match_reg is not None:
            present_reg_nos.add(best_match_reg)
            if best_match_reg not in matches_info or best_sim > matches_info[best_match_reg]:
                matches_info[best_match_reg] = best_sim
            present_confidences.append(best_sim)
            
            # Green bounding box for recognized student
            yolo_tag = f" | YOLOv8:{best_yolo_conf:.2f}" if best_yolo_conf is not None else ""
            label = f"{best_student['name']} ({best_match_reg}) [Sim:{best_sim:.2f}{yolo_tag}]"
            cv2.rectangle(annotated_img, (box_to_draw[0], box_to_draw[1]), (box_to_draw[2], box_to_draw[3]), (0, 255, 0), 2)
            
            # Background banner for label
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(annotated_img, (box_to_draw[0], max(0, box_to_draw[1] - 18)), (box_to_draw[0] + w, box_to_draw[1]), (0, 255, 0), -1)
            cv2.putText(annotated_img, label, (box_to_draw[0], max(10, box_to_draw[1] - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
        else:
            # Unknown / Unmatched face (Red bounding box)
            yolo_tag = f" | YOLOv8:{best_yolo_conf:.2f}" if best_yolo_conf is not None else ""
            sim_str = f" [{best_sim:.2f}{yolo_tag}]" if best_sim > 0 else (f" [{yolo_tag.strip(' | ')}]" if yolo_tag else "")
            label = f"Unknown{sim_str}"
            cv2.rectangle(annotated_img, (box_to_draw[0], box_to_draw[1]), (box_to_draw[2], box_to_draw[3]), (0, 0, 255), 2)
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(annotated_img, (box_to_draw[0], max(0, box_to_draw[1] - 18)), (box_to_draw[0] + w, box_to_draw[1]), (0, 0, 255), -1)
            cv2.putText(annotated_img, label, (box_to_draw[0], max(10, box_to_draw[1] - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    # 3. Highlight any YOLO-detected persons where face was partially obscured/unextracted
    for idx, y_det in enumerate(yolo_detections):
        if idx not in matched_yolo_indices:
            y_box = y_det["bbox"]
            cv2.rectangle(annotated_img, (y_box[0], y_box[1]), (y_box[2], y_box[3]), (255, 191, 0), 1)
            y_lbl = f"YOLO Person [{y_det['confidence']:.2f}]"
            cv2.putText(annotated_img, y_lbl, (y_box[0], max(12, y_box[1] - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 191, 0), 1)

    # 4. Stamp GPS Geolocation Banner directly onto the processed photo
    cur_stamp_time = timestamp_str or datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    annotated_img = location_utils.stamp_gps_on_image(
        image_bgr=annotated_img,
        location_data=location_data,
        slot_name=slot_name,
        timestamp_str=cur_stamp_time
    )

    # Build attendance list
    attendance_records = []
    for student in registered_students:
        reg = student["register_no"]
        is_present = reg in present_reg_nos
        attendance_records.append({
            "Register No": reg,
            "Name": student["name"],
            "Department": student["department"],
            "Status": "Present" if is_present else "Absent",
            "Match Confidence": f"{matches_info[reg]:.2f}" if reg in matches_info else "N/A"
        })
        
    p_cnt = len(present_reg_nos)
    tot_reg = len(registered_students)
    avg_conf = round(float(np.mean(present_confidences)), 3) if present_confidences else None
    
    # Genuine calculated attendance accuracy rate
    genuine_accuracy_pct = round((p_cnt / max(tot_reg, 1)) * 100, 1)

    metrics = {
        "total_registered": tot_reg,
        "total_detected": len(detected_faces),
        "yolo_persons_detected": len(yolo_detections),
        "present": p_cnt,
        "absent": tot_reg - p_cnt,
        "attendance_rate_pct": genuine_accuracy_pct,
        "avg_match_confidence": avg_conf,
        "yolo_model": yolo_model_name,
        "yolo_speed": yolo_speed
    }

    # 5. Print comprehensive YOLO accuracy details in terminal
    print_yolo_terminal_accuracy_report(
        model_name=yolo_model_name,
        yolo_detections=yolo_detections,
        speed_metrics=yolo_speed,
        detected_faces_count=len(detected_faces),
        present_count=p_cnt,
        total_students=tot_reg,
        avg_face_confidence=avg_conf,
        location_data=location_data,
        slot_name=slot_name
    )
    
    return annotated_img, attendance_records, metrics

