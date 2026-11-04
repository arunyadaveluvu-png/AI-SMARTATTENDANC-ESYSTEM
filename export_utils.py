import io
import numpy as np
import pandas as pd
from PIL import Image as PILImage
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

def create_pdf_report(
    slot_name: str,
    faculty_name: str,
    date_str: str,
    time_str: str,
    total_strength: int,
    present_count: int,
    absent_count: int,
    df_records: pd.DataFrame,
    annotated_images: list = None,
    location_str: str = None,
    yolo_model_name: str = None
) -> bytes:
    """
    Generate an official attendance PDF report containing ONLY present students
    with header metadata block (Slot, Faculty Name, Date, Time, Total Strength, Present Count, Absent Count, GPS Location, YOLO Model)
    and embedded captured/uploaded photos with bounding box evidence at the bottom.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )
    
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=colors.HexColor('#0f172a'),
        alignment=1, # Center
        spaceAfter=4
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=14,
        textColor=colors.HexColor('#0284c7'),
        alignment=1,
        spaceAfter=10
    )

    meta_label_style = ParagraphStyle(
        'MetaLabel',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9.5,
        leading=12,
        textColor=colors.HexColor('#334155')
    )
    
    meta_val_style = ParagraphStyle(
        'MetaVal',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=12,
        textColor=colors.HexColor('#0f172a')
    )
    
    tbl_hdr_style = ParagraphStyle(
        'TblHdr',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=12,
        textColor=colors.white,
        alignment=1
    )
    
    tbl_cell_style = ParagraphStyle(
        'TblCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=12,
        textColor=colors.HexColor('#1e293b')
    )

    story = []
    
    # Title Header
    story.append(Paragraph("AI FACE RECOGNITION ATTENDANCE SYSTEM", title_style))
    story.append(Paragraph("Official Class Attendance Report (Present Students)", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#0284c7'), spaceAfter=10))
    
    # Metadata Block Table
    loc_display = location_str if (location_str and location_str.strip()) else "Not Permitted / Unavailable"
    if yolo_model_name and "yolov8" in yolo_model_name.lower():
        model_display = f"YOLOv8 ({yolo_model_name})"
    else:
        model_display = yolo_model_name if (yolo_model_name and yolo_model_name.strip()) else "YOLOv8 Nano (Default Model)"
    
    meta_data = [
        [
            Paragraph("<b>Slot / Session:</b>", meta_label_style), Paragraph(slot_name, meta_val_style),
            Paragraph("<b>Faculty In-Charge:</b>", meta_label_style), Paragraph(faculty_name, meta_val_style),
            Paragraph("<b>Date & Time:</b>", meta_label_style), Paragraph(f"{date_str} {time_str}", meta_val_style)
        ],
        [
            Paragraph("<b>Total Strength:</b>", meta_label_style), Paragraph(str(total_strength), meta_val_style),
            Paragraph("<b>Present Count:</b>", meta_label_style), Paragraph(f"<font color='#15803d'><b>{present_count}</b></font>", meta_val_style),
            Paragraph("<b>Absent Count:</b>", meta_label_style), Paragraph(f"<font color='#b91c1c'><b>{absent_count}</b></font>", meta_val_style)
        ],
        [
            Paragraph("<b>GPS Location:</b>", meta_label_style), Paragraph(loc_display, meta_val_style),
            Paragraph("<b>AI Model Engine:</b>", meta_label_style), Paragraph(model_display, meta_val_style),
            Paragraph("", meta_label_style), Paragraph("", meta_val_style)
        ]
    ]
    
    meta_table = Table(meta_data, colWidths=[90, 110, 95, 115, 60, 50])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f1f5f9')),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#cbd5e1')),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e2e8f0')),
        ('SPAN', (1, 2), (1, 2)), # Location
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 5),
        ('RIGHTPADDING', (0,0), (-1,-1), 5),
    ]))
    
    story.append(meta_table)
    story.append(Spacer(1, 10))
    
    # Filter ONLY Present Students
    df_present = df_records[df_records["Status"] == "Present"].reset_index(drop=True)
    
    # Attendance Roster Table (Present Students Only)
    table_rows = [
        [
            Paragraph("S.No", tbl_hdr_style),
            Paragraph("Registration Number", tbl_hdr_style),
            Paragraph("Student Name", tbl_hdr_style),
            Paragraph("Department", tbl_hdr_style),
            Paragraph("Status", tbl_hdr_style)
        ]
    ]
    
    if not df_present.empty:
        for idx, row in df_present.iterrows():
            s_no = str(idx + 1)
            reg = str(row.get("Register No", ""))
            name = str(row.get("Name", ""))
            dept = str(row.get("Department", ""))
            
            status_p = Paragraph("<font color='#15803d'><b>Present</b></font>", tbl_cell_style)
            
            table_rows.append([
                Paragraph(s_no, tbl_cell_style),
                Paragraph(reg, tbl_cell_style),
                Paragraph(name, tbl_cell_style),
                Paragraph(dept, tbl_cell_style),
                status_p
            ])
    else:
        table_rows.append([
            Paragraph("1", tbl_cell_style),
            Paragraph("N/A", tbl_cell_style),
            Paragraph("No Present Students Recorded", tbl_cell_style),
            Paragraph("N/A", tbl_cell_style),
            Paragraph("<font color='#b91c1c'><b>None</b></font>", tbl_cell_style)
        ])
        
    roster_table = Table(table_rows, colWidths=[35, 120, 160, 135, 70])
    
    # Apply styling
    t_style = [
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0284c7')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
    ]
    
    for r in range(1, len(table_rows)):
        bg_color = colors.HexColor('#f8fafc') if r % 2 == 0 else colors.white
        t_style.append(('BACKGROUND', (0, r), (-1, r), bg_color))
        
    roster_table.setStyle(TableStyle(t_style))
    story.append(roster_table)
    
    # Embed Captured / Uploaded Group Photos at the Bottom
    if annotated_images:
        story.append(Spacer(1, 16))
        section_style = ParagraphStyle(
            'SectionTitle',
            parent=styles['Heading2'],
            fontName='Helvetica-Bold',
            fontSize=12,
            leading=15,
            textColor=colors.HexColor('#0284c7'),
            spaceAfter=6
        )
        story.append(Paragraph("📷 Captured AI Visual Face Recognition Photos", section_style))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#cbd5e1'), spaceAfter=10))
        
        img_caption_style = ParagraphStyle(
            'ImgCaption',
            parent=styles['Normal'],
            fontName='Helvetica-Oblique',
            fontSize=8.5,
            leading=11,
            textColor=colors.HexColor('#64748b'),
            alignment=1,
            spaceAfter=10
        )

        for idx, img_item in enumerate(annotated_images):
            try:
                if isinstance(img_item, np.ndarray):
                    pil_img = PILImage.fromarray(img_item)
                elif isinstance(img_item, PILImage.Image):
                    pil_img = img_item
                else:
                    pil_img = PILImage.open(io.BytesIO(img_item))
                    
                img_buf = io.BytesIO()
                pil_img.save(img_buf, format="JPEG", quality=80)
                img_buf.seek(0)
                
                # Scale image ensuring both width and height fit safely within page frame
                w, h = pil_img.size
                max_w = 460.0
                max_h = 360.0
                
                scale = min(max_w / max(w, 1), max_h / max(h, 1))
                target_w = max(1, int(w * scale))
                target_h = max(1, int(h * scale))
                
                rl_img = RLImage(img_buf, width=target_w, height=target_h)
                story.append(rl_img)
                story.append(Spacer(1, 4))
                gps_caption_suffix = f" | GPS Watermark: {loc_display}" if loc_display else ""
                story.append(Paragraph(f"Classroom Photo #{idx + 1} - AI Face Recognition & Evidence{gps_caption_suffix}", img_caption_style))
                story.append(Spacer(1, 10))
            except Exception as ex:
                pass

                
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


def create_excel_report(
    slot_name: str,
    faculty_name: str,
    date_str: str,
    time_str: str,
    total_strength: int,
    present_count: int,
    absent_count: int,
    df_records: pd.DataFrame,
    location_str: str = None,
    yolo_model_name: str = None
) -> bytes:
    """
    Generate an Excel (.xlsx) file containing ONLY present students
    with header metadata block (Slot, Faculty Name, Date, Time, Total Strength, Present Count, Absent Count, GPS Location, YOLO Model).
    """
    buffer = io.BytesIO()
    loc_display = location_str if (location_str and location_str.strip()) else "Not Permitted / Unavailable"
    model_display = yolo_model_name if (yolo_model_name and yolo_model_name.strip()) else "YOLOv8n Deep Learning"
    
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        # Create Summary Metadata Header
        meta_rows = [
            ["AI FACE RECOGNITION ATTENDANCE SYSTEM - OFFICIAL REPORT", ""],
            ["Slot / Session", slot_name],
            ["Faculty In-Charge", faculty_name],
            ["Date", date_str],
            ["Time", time_str],
            ["Total Strength", total_strength],
            ["Present Count", present_count],
            ["Absent Count", absent_count],
            ["GPS Location", loc_display],
            ["AI Model Engine", model_display],
            ["", ""] # Empty row separator
        ]
        
        meta_df = pd.DataFrame(meta_rows, columns=["Parameter", "Value"])
        meta_df.to_excel(writer, sheet_name="Attendance Sheet", index=False, startrow=0)
        
        # Prepare ONLY Present Students Table with S.No
        df_present = df_records[df_records["Status"] == "Present"].copy().reset_index(drop=True)
        
        if not df_present.empty:
            df_present.insert(0, "S.No", range(1, len(df_present) + 1))
            if "Register No" in df_present.columns:
                df_present.rename(columns={"Register No": "Registration Number", "Name": "Student Name"}, inplace=True)
        else:
            df_present = pd.DataFrame([{
                "S.No": 1,
                "Registration Number": "N/A",
                "Student Name": "No Present Students Recorded",
                "Department": "N/A",
                "Status": "None"
            }])
            
        start_row_table = len(meta_rows) + 2
        df_present.to_excel(writer, sheet_name="Attendance Sheet", index=False, startrow=start_row_table)

    buffer.seek(0)
    return buffer.getvalue()
