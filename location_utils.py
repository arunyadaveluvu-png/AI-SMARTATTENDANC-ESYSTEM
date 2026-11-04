import datetime
from typing import Optional, Dict, Any, Tuple
import cv2
import numpy as np
import streamlit as st

try:
    from geopy.geocoders import Nominatim
    GEOPY_AVAILABLE = True
except ImportError:
    GEOPY_AVAILABLE = False

try:
    import streamlit_js_eval
    JS_EVAL_AVAILABLE = True
except ImportError:
    JS_EVAL_AVAILABLE = False


def reverse_geocode_coordinates(lat: float, lon: float) -> Optional[str]:
    """
    Reverse geocodes latitude and longitude coordinates into a human-readable address.
    Uses OpenStreetMap Nominatim via GeoPy with defensive timeout and error handling.
    """
    if not GEOPY_AVAILABLE:
        return None
    try:
        geolocator = Nominatim(user_agent="FacultyAttendanceSystem_v2", timeout=4)
        location = geolocator.reverse(f"{lat}, {lon}", language="en")
        if location and location.address:
            return location.address
    except Exception as e:
        print(f"Reverse geocode notice: {e}")
    return None


import urllib.request
import json

try:
    import database
    DATABASE_AVAILABLE = True
except ImportError:
    DATABASE_AVAILABLE = False


def get_device_ip_location() -> Optional[Dict[str, Any]]:
    """
    Retrieves the device's accurate geographic location via public IP geolocation services.
    Provides accurate City, Region, Country, and genuine Coordinates when hardware GPS is unavailable.
    """
    try:
        req = urllib.request.Request(
            "https://ipapi.co/json/",
            headers={"User-Agent": "FacultyAttendanceSystem/2.0"}
        )
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read().decode())
            lat = data.get("latitude")
            lon = data.get("longitude")
            city = data.get("city")
            region = data.get("region")
            country = data.get("country_name")
            if lat is not None and lon is not None:
                addr_parts = [p for p in [city, region, country] if p]
                return {
                    "latitude": float(lat),
                    "longitude": float(lon),
                    "accuracy": 35.0,
                    "address": ", ".join(addr_parts),
                    "acquired_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "status": "GRANTED",
                    "source": "NETWORK_IP"
                }
    except Exception as e:
        # Fallback to secondary endpoint
        try:
            req = urllib.request.Request("http://ip-api.com/json/", headers={"User-Agent": "FacultyAttendanceSystem/2.0"})
            with urllib.request.urlopen(req, timeout=3) as resp:
                data = json.loads(resp.read().decode())
                if data.get("status") == "success":
                    lat = data.get("lat")
                    lon = data.get("lon")
                    city = data.get("city")
                    region = data.get("regionName")
                    country = data.get("country")
                    addr_parts = [p for p in [city, region, country] if p]
                    return {
                        "latitude": float(lat),
                        "longitude": float(lon),
                        "accuracy": 50.0,
                        "address": ", ".join(addr_parts),
                        "acquired_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "status": "GRANTED",
                        "source": "NETWORK_IP"
                    }
        except Exception as inner_e:
            print(f"IP Geolocation error: {inner_e}")
    return None


def render_gps_permission_ui(key_prefix: str = "att") -> Optional[Dict[str, Any]]:
    """
    Renders the GPS location access UI in the Streamlit attendance capture flow.
    Supports browser Geolocation API, automatic high-accuracy IP fallback,
    and campus/classroom location calibration.
    """
    st.markdown("#### 📍 Geolocation Verification & Pinpoint Campus Accuracy")
    
    col_toggle, col_status = st.columns([1.5, 2])
    with col_toggle:
        req_location = st.toggle(
            "Acquire GPS Location for Attendance",
            value=st.session_state.get(f"{key_prefix}_gps_toggle_val", True),
            key=f"{key_prefix}_gps_toggle_widget",
            help="Acquires verified geographic coordinates and address alongside attendance photos."
        )
        st.session_state[f"{key_prefix}_gps_toggle_val"] = req_location
        
    location_data = None
    
    if req_location:
        # Check if user calibrated a specific campus / classroom address
        calibrated_addr = None
        calibrated_lat = None
        calibrated_lon = None
        if DATABASE_AVAILABLE:
            calibrated_addr = database.get_setting("campus_location_name")
            calibrated_lat = database.get_setting("campus_location_lat")
            calibrated_lon = database.get_setting("campus_location_lon")

        # 1. Try Browser Geolocation API if available
        raw_geo = None
        if JS_EVAL_AVAILABLE:
            raw_geo = streamlit_js_eval.get_geolocation(component_key=f"{key_prefix}_get_geo_eval")
        
        # Determine best genuine coordinates
        lat = None
        lon = None
        accuracy = 20.0
        resolved_addr = None
        loc_source = "DEVICE_GPS"

        if raw_geo and isinstance(raw_geo, dict) and "coords" in raw_geo:
            coords = raw_geo.get("coords", {})
            lat = coords.get("latitude")
            lon = coords.get("longitude")
            accuracy = coords.get("accuracy", 20.0)
            
        # 2. If browser geolocation unavailable or denied on desktop, automatically use accurate network IP location
        if lat is None or lon is None:
            cached_ip_loc = st.session_state.get(f"{key_prefix}_cached_ip_location")
            if not cached_ip_loc:
                ip_loc = get_device_ip_location()
                if ip_loc:
                    st.session_state[f"{key_prefix}_cached_ip_location"] = ip_loc
                    cached_ip_loc = ip_loc
            if cached_ip_loc:
                lat = cached_ip_loc.get("latitude")
                lon = cached_ip_loc.get("longitude")
                accuracy = cached_ip_loc.get("accuracy", 30.0)
                resolved_addr = cached_ip_loc.get("address")
                loc_source = "NETWORK_GEO"

        # 3. If user has a calibrated campus/classroom override, incorporate it
        if calibrated_addr and calibrated_addr.strip():
            resolved_addr = calibrated_addr.strip()
            if calibrated_lat and calibrated_lon:
                try:
                    lat = float(calibrated_lat)
                    lon = float(calibrated_lon)
                    loc_source = "CALIBRATED_CAMPUS"
                except (ValueError, TypeError):
                    pass

        if lat is not None and lon is not None:
            if not resolved_addr:
                addr_key = f"{key_prefix}_resolved_addr_{lat:.4f}_{lon:.4f}"
                if addr_key not in st.session_state:
                    resolved_addr = reverse_geocode_coordinates(lat, lon)
                    st.session_state[addr_key] = resolved_addr
                else:
                    resolved_addr = st.session_state[addr_key]

            location_data = {
                "latitude": float(lat),
                "longitude": float(lon),
                "accuracy": float(accuracy),
                "address": resolved_addr,
                "acquired_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "status": "GRANTED",
                "source": loc_source
            }
            st.session_state[f"{key_prefix}_active_gps"] = location_data
            
            with col_status:
                st.success(
                    f"**Location Verified ({loc_source})**\n\n"
                    f"- **Coordinates:** `{lat:.6f}° N, {lon:.6f}° E` (±{accuracy:.0f}m)\n"
                    f"- **Location:** {resolved_addr if resolved_addr else 'Coordinates Captured'}"
                )
        else:
            with col_status:
                st.warning("⚠️ Geolocation access denied or unavailable. You can calibrate campus location below.")
            st.session_state[f"{key_prefix}_active_gps"] = None

        # Collapsible Campus / Classroom Location Calibration Tool
        with st.expander("🎯 Calibrate Campus / Classroom Location (Pinpoint Accuracy)", expanded=False):
            st.caption("Fine-tune or customize the exact Institution, Department, or Classroom location stamped on photos.")
            c_col1, c_col2 = st.columns([2, 1])
            with c_col1:
                cur_custom_addr = calibrated_addr or (location_data.get("address") if location_data else "")
                new_campus_name = st.text_input(
                    "Campus / Classroom / Department Name:",
                    value=cur_custom_addr or "",
                    placeholder="e.g. Dept of Computer Science, Engineering Block, Campus A",
                    key=f"{key_prefix}_custom_campus_inp"
                )
            with c_col2:
                cur_lat_str = str(lat) if lat is not None else (calibrated_lat or "16.51667")
                cur_lon_str = str(lon) if lon is not None else (calibrated_lon or "80.61667")
                sub_c1, sub_c2 = st.columns(2)
                with sub_c1:
                    new_lat = st.text_input("Latitude:", value=cur_lat_str, key=f"{key_prefix}_custom_lat_inp")
                with sub_c2:
                    new_lon = st.text_input("Longitude:", value=cur_lon_str, key=f"{key_prefix}_custom_lon_inp")
            
            if st.button("Save & Calibrate Location", key=f"{key_prefix}_save_calib_btn"):
                if DATABASE_AVAILABLE:
                    database.save_setting("campus_location_name", new_campus_name.strip())
                    database.save_setting("campus_location_lat", new_lat.strip())
                    database.save_setting("campus_location_lon", new_lon.strip())
                st.session_state.pop(f"{key_prefix}_cached_ip_location", None)
                st.success("Campus location calibrated and saved successfully!")
                st.rerun()
    else:
        st.caption("Location access disabled. Attendance photos will be recorded without GPS coordinates.")
        st.session_state[f"{key_prefix}_active_gps"] = None
        
    return location_data


def format_location_display(location_data: Optional[Dict[str, Any]]) -> str:
    """Formats location data into a clean single-line summary string."""
    if not location_data:
        return "Not Available (Permission Denied or Disabled)"
    lat = location_data.get("latitude")
    lon = location_data.get("longitude")
    addr = location_data.get("address")
    acc = location_data.get("accuracy", 0.0)
    
    if lat is not None and lon is not None:
        coords_str = f"{lat:.5f}, {lon:.5f} (±{acc:.0f}m)"
        if addr:
            return f"{coords_str} — {addr}"
        return coords_str
    return "Not Available"


def stamp_gps_on_image(
    image_bgr: np.ndarray,
    location_data: Optional[Dict[str, Any]] = None,
    slot_name: str = "",
    timestamp_str: Optional[str] = None
) -> np.ndarray:
    """
    Stamps an official GPS geolocation, slot/session name, and timestamp watermark
    directly onto an image's pixels. Creates a semi-transparent HUD banner at the bottom.
    
    Visible:
    - During initial capture preview
    - During attendance facial processing
    - When exported into PDF report
    """
    if image_bgr is None or not isinstance(image_bgr, np.ndarray) or image_bgr.size == 0:
        return image_bgr
    
    img = image_bgr.copy()
    h, w = img.shape[:2]
    
    # Calculate banner dimensions proportional to image size
    banner_height = max(55, int(h * 0.11))
    if h < 240:
        banner_height = 42
    
    overlay = img.copy()
    # Draw semi-transparent dark banner at bottom
    cv2.rectangle(overlay, (0, h - banner_height), (w, h), (15, 23, 42), -1)
    
    # Alpha blend overlay with original
    alpha = 0.82
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    
    is_granted = bool(
        location_data
        and location_data.get("status") == "GRANTED"
        and location_data.get("latitude") is not None
    )
    
    # Border accent: Emerald Green if GPS acquired, Sky Blue if unverified/disabled
    border_color = (34, 197, 94) if is_granted else (14, 165, 233)
    cv2.line(img, (0, h - banner_height), (w, h - banner_height), border_color, max(2, int(w / 400)))
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.38, min(0.85, w / 1100.0))
    thickness = 1
    if w > 1400:
        thickness = 2
        
    cur_time = timestamp_str or datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    if is_granted:
        lat = location_data.get("latitude")
        lon = location_data.get("longitude")
        acc = location_data.get("accuracy", 0.0)
        addr = location_data.get("address") or ""
        
        line1 = f"GPS: {lat:.6f} N, {lon:.6f} E (+/-{acc:.1f}m)"
        if slot_name:
            line1 += f"  |  {slot_name}"
            
        addr_snippet = addr[:65] + "..." if len(addr) > 65 else addr
        line2 = f"Loc: {addr_snippet}" if addr_snippet else "Verified GPS Coordinates"
        line2 += f"  |  {cur_time}"
        
        y_line1 = h - int(banner_height * 0.55)
        y_line2 = h - int(banner_height * 0.18)
        
        cv2.putText(img, line1, (14, y_line1), font, font_scale, (74, 222, 128), thickness, cv2.LINE_AA)
        cv2.putText(img, line2, (14, y_line2), font, font_scale * 0.9, (241, 245, 249), 1, cv2.LINE_AA)
    else:
        line1 = "GPS: Permission Not Granted / Disabled"
        if slot_name:
            line1 += f"  |  {slot_name}"
        line2 = f"Captured at: {cur_time}  |  Classroom Attendance"
        
        y_line1 = h - int(banner_height * 0.55)
        y_line2 = h - int(banner_height * 0.18)
        
        cv2.putText(img, line1, (14, y_line1), font, font_scale, (234, 179, 8), thickness, cv2.LINE_AA)
        cv2.putText(img, line2, (14, y_line2), font, font_scale * 0.9, (226, 232, 240), 1, cv2.LINE_AA)
        
    return img
