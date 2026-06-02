# File: app_cam_yolo_area.py
"""
PyQt6 YOLO viewer with Area Calculation & Screen Capture (CPU ONLY).

Requirements:
    pip install pyqt6 opencv-python ultralytics torch torchvision sahi

Usage:
    python app_cam_yolo_area.py
"""

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QComboBox, QVBoxLayout,
    QHBoxLayout, QPushButton, QSizePolicy, QFrame, QGroupBox,
    QCheckBox, QScrollArea, QProgressBar, QSlider
)
from PyQt6.QtGui import QImage, QPixmap, QFont, QPainter, QPen, QColor, QGuiApplication, QIcon
from PyQt6.QtCore import Qt, pyqtSignal, QRect, QRectF, QPoint, QPointF
import sys
import cv2
import numpy as np
import os
import json
import time

# Try to import ultralytics YOLO
try:
    from ultralytics import YOLO
except Exception as e:
    YOLO = None
    print(f"Ultralytics import error: {e}")

# Try to import SAHI for tiled inference (Using standard stable imports)
try:
    from sahi.predict import get_sliced_prediction
    from sahi import AutoDetectionModel
    HAS_SAHI = True
except Exception as e:
    HAS_SAHI = False
    print('SAHI Setup Error:', e)

# ---- CONFIG ----
# Provide the YOLO model paths pointing to 'essential_files' directory
MODEL_PATHS = {
    "KI67 analysis": os.path.join("essential_files", "yolo_KI67.pt"),
    "ER/PR analysis": os.path.join("essential_files", "yolo_ER.pt"),
}

CLASS_NAMES = {
    "KI67 analysis": ["Positive cells", "Negative cells"],
    "ER/PR analysis": ["Positive cells", "Negative cells"],
}

INFERENCE_SIZE = 640  
CONF_THRESHOLD = 0.25

# Strictly enforce CPU mode
DEVICE = "cpu"

# ---- MODERN STYLESHEET ----
MODERN_QSS_TEMPLATE = """
QMainWindow {
    background-color: #f1f5f9;
}

QScrollArea#SidePanel {
    background-color: #ffffff;
    border-radius: 12px;
    border: 1px solid #e2e8f0;
}

QWidget#SidePanelInner {
    background-color: transparent;
}

/* Custom modern scrollbar for the sidebar */
QScrollBar:vertical {
    border: none;
    background: transparent;
    width: 8px;
    margin: 0px 0px 0px 0px;
    border-radius: 4px;
}
QScrollBar::handle:vertical {
    background: #cbd5e1;
    min-height: 20px;
    border-radius: 4px;
}
QScrollBar::handle:vertical:hover {
    background: #94a3b8;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}

QFrame#VideoPanel {
    background-color: #ffffff;
    border-radius: 12px;
    border: 1px solid #e2e8f0;
}

QGroupBox {
    font-weight: bold;
    color: #1e3a8a;
    border: 1px solid #cbd5e1;
    border-radius: 8px;
    margin-top: 18px; 
    padding-top: 18px; 
    background-color: #ffffff;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 12px;
    top: 0px; 
    background-color: #ffffff; 
    padding: 0 6px;
}

QLabel {
    color: #334155;
}

QWidget#VideoScreen {
    background-color: #0f172a;
    border-radius: 8px;
}

QPushButton {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #3b82f6, stop:1 #2563eb);
    color: white;
    border: none;
    border-radius: 6px;
    padding: 8px 12px;
    font-weight: bold;
}

QPushButton:hover {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #60a5fa, stop:1 #3b82f6);
}

QPushButton:pressed {
    background: #1d4ed8;
}

QPushButton:disabled {
    background: #94a3b8;
}

QPushButton:checked {
    background: #1e3a8a;
    border: 2px solid #bfdbfe;
}

QComboBox {
    border: 1px solid #cbd5e1;
    border-radius: 6px;
    padding: 6px 30px 6px 10px;
    background: #f8fafc;
    color: #0f172a;
    font-weight: 500;
}

QComboBox::drop-down {
    subcontrol-origin: padding;
    subcontrol-position: top right;
    width: 30px;
    border-left: none;
}

QComboBox::down-arrow {
    image: url("__ARROW_URL__");
    width: 14px;
    height: 14px;
}

QComboBox:hover {
    border: 1px solid #94a3b8;
}

QCheckBox {
    color: #334155;
    font-weight: 500;
}

QCheckBox::indicator {
    width: 16px;
    height: 16px;
}
"""

def get_stylesheet():
    """Generates the native PNG icon dynamically to avoid missing SVG plugin issues."""
    os.makedirs("essential_files", exist_ok=True)
    arrow_path = os.path.abspath(os.path.join("essential_files", "custom_arrow.png"))
    
    if not os.path.exists(arrow_path):
        pixmap = QPixmap(32, 32)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        pen = QPen(QColor("#334155"))
        pen.setWidth(4)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        # Draw a nice centered chevron
        painter.drawPolyline([QPoint(8, 12), QPoint(16, 20), QPoint(24, 12)])
        painter.end()
        pixmap.save(arrow_path, "PNG")

    arrow_url = arrow_path.replace("\\", "/")
    return MODERN_QSS_TEMPLATE.replace("__ARROW_URL__", arrow_url)


# -----------------
# Image Enhancement Functions
# -----------------
def apply_macenko_calibration(img_bgr):
    """Lighter auto-white balance mimicking Macenko's background correction."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_flat = img_rgb.reshape(-1, 3)
    bg_color = np.percentile(img_flat, 95, axis=0)
    bg_color = np.maximum(bg_color, 10.0) # Avoid division by zero
    
    img_norm = np.clip((img_rgb / bg_color) * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(img_norm, cv2.COLOR_RGB2BGR)

def apply_vahadane(img_bgr, alpha=0.10):
    """Structure-preserving stain normalization mapped to stable target statistics."""
    target_HE = np.array([[0.5626, 0.2159], [0.7201, 0.8012], [0.4062, 0.5581]])
    target_max_C = np.array([1.9705, 1.0308])

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = img_rgb.astype(np.float32) + 1 
    Io = 255.0
    OD = -np.log(img_rgb / Io)
    
    OD_hat = OD[~np.any(OD < 0.15, axis=2)]
    if OD_hat.shape[0] < 100:
        return img_bgr 
    
    _, eigvecs = np.linalg.eigh(np.cov(OD_hat.T))
    eigvecs = eigvecs[:, [1, 2]]
    
    T_hat = np.dot(OD_hat, eigvecs)
    phi = np.arctan2(T_hat[:, 1], T_hat[:, 0])
    minPhi = np.percentile(phi, 1)
    maxPhi = np.percentile(phi, 99)
    
    vMin = np.dot(eigvecs, np.array([np.cos(minPhi), np.sin(minPhi)]))
    vMax = np.dot(eigvecs, np.array([np.cos(maxPhi), np.sin(maxPhi)]))
    
    HE = np.array([vMin, vMax]).T if vMin[0] > vMax[0] else np.array([vMax, vMin]).T
    HE = HE / np.linalg.norm(HE, axis=0)[None, :]
    
    C = np.linalg.lstsq(HE, OD.reshape((-1, 3)).T, rcond=None)[0]
    max_C = np.percentile(C, 99, axis=1)
    max_C = np.maximum(max_C, 1e-6)
    
    C *= (target_max_C / max_C)[:, None]
    
    norm_OD = np.dot(target_HE, C).T.reshape(img_rgb.shape)
    norm_rgb = Io * np.exp(-norm_OD)
    norm_rgb = np.clip(norm_rgb, 0, 255).astype(np.uint8)
    norm_bgr = cv2.cvtColor(norm_rgb, cv2.COLOR_RGB2BGR)
    
    if alpha >= 1.0:
        return norm_bgr
    return cv2.addWeighted(norm_bgr, alpha, img_bgr, 1.0 - alpha, 0)

def apply_clahe(img_bgr, clip_limit=0.5):
    """Applies Contrast Limited Adaptive Histogram Equalization to the L channel."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=max(0.1, float(clip_limit)), tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

def apply_unsharp_mask(img_bgr, kernel_size=(5, 5), sigma=1.0, amount=0.5, threshold=0):
    """Applies an unsharp mask to enhance edges."""
    blurred = cv2.GaussianBlur(img_bgr, kernel_size, sigma)
    sharpened = float(amount + 1) * img_bgr - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(img_bgr - blurred) < threshold
        np.copyto(sharpened, img_bgr, where=low_contrast_mask)
    return sharpened


# -----------------
# Helper Function for YOLO SAHI Inference & Localized Shape Filtering
# -----------------
def run_screenshot_inference(img_rgb, model, current_option, progress_callback=None, invert_classes=False,
                           use_macenko=True, use_vahadane=True, vahadane_val=0.10,
                           use_clahe=True, clahe_val=0.5, use_unsharp=True, unsharp_val=0.5,
                           include_all_cells=False, size_pct_thresh=0.30, circ_thresh=0.40, round_thresh=0.40):
    
    boxes_out = []
    
    try:
        if progress_callback:
            progress_callback(5, "Enhancing image and loading YOLO_SAHI...")
            
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        
        # --- Apply sequential pre-processing enhancements ---
        if use_macenko:
            if progress_callback: progress_callback(8, "Auto-calibrating background...")
            img_bgr = apply_macenko_calibration(img_bgr)
            
        if use_vahadane:
            if progress_callback: progress_callback(12, "Applying Vahadane Stain Normalization...")
            img_bgr = apply_vahadane(img_bgr, alpha=vahadane_val)

        if use_clahe:
            if progress_callback: progress_callback(16, "Applying CLAHE enhancement...")
            img_bgr = apply_clahe(img_bgr, clip_limit=clahe_val)
            
        if use_unsharp:
            if progress_callback: progress_callback(20, "Applying Unsharp Masking...")
            img_bgr = apply_unsharp_mask(img_bgr, amount=unsharp_val)

        # Pre-compute DAB channel globally for ER/PR intensity calculation AND KI67 Override
        if current_option in ["ER/PR analysis", "KI67 analysis"]:
            if progress_callback: progress_callback(30, "Calculating global DAB Intensities...")
            HDAB_matrix = np.array([
                [0.644, 0.716, 0.266], # Hematoxylin
                [0.268, 0.570, 0.776], # DAB
                [0.711, 0.423, 0.561]  # Residual
            ], dtype=np.float32)
            HDAB_inv = np.linalg.inv(HDAB_matrix)
            
            img_rgb_float = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
            img_flat = img_rgb_float.reshape(-1, 3)
            bg_color = np.percentile(img_flat, 95, axis=0)
            bg_color = np.maximum(bg_color, 10.0)
            img_norm = (img_rgb_float + 1.0) / bg_color
            OD = -np.log10(np.clip(img_norm, 1e-4, 1.0))
            C = np.dot(OD, HDAB_inv)
            dab_channel = C[:, :, 1]
            
        # The returned image will be solely the processed RGB (no boxes drawn natively)
        drawn_blend_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        if progress_callback: progress_callback(40, "Running Tiled Inference (SAHI)...")

        # --- EVALUATE BOUNDING BOXES ---
        results = get_sliced_prediction(
            image=img_bgr, 
            detection_model=model,
            slice_height=640,
            slice_width=640,
            overlap_height_ratio=0.15,
            overlap_width_ratio=0.15,
            verbose=False
        )
        
        if progress_callback: progress_callback(70, "Filtering and analyzing cell morphology...")
        
        raw_objects = []
        pos_areas = []
        
        # Pass 1: Collect boxes and compute areas for median analysis
        for obj in results.object_prediction_list:
            x1, y1 = int(obj.bbox.minx), int(obj.bbox.miny)
            x2, y2 = int(obj.bbox.maxx), int(obj.bbox.maxy)
            
            # Use original exact SAHI mapping logic:
            raw_cls_id = int(obj.category.id)
            cls_id = 1 - raw_cls_id
            
            if invert_classes:
                cls_id = 1 - cls_id
            
            box_area = (x2 - x1) * (y2 - y1)
            
            if cls_id == 0:  # 0 is strictly Positive across all apps
                pos_areas.append(box_area)
                
            raw_objects.append({'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'cls_id': cls_id, 'area': box_area})
            
        median_pos_area = np.median(pos_areas) if pos_areas else 0
        min_area_thresh = median_pos_area * size_pct_thresh
        
        H, W, _ = img_bgr.shape
        
        # Pass 2: Filter by Relative Size, Cellularity, and Roundness, AND Override DAB
        for idx, obj in enumerate(raw_objects):
            if progress_callback and idx % 20 == 0:
                progress_callback(70 + int(25 * (idx / max(1, len(raw_objects)))), "Filtering shapes...")

            x1, y1, x2, y2 = obj['x1'], obj['y1'], obj['x2'], obj['y2']
            cls_id = obj['cls_id']
            box_area = obj['area']
            
            mask_valid = False
            skip_cell = False
            
            # Compute a cell mask if required for Shape Filtering OR ER/PR/KI67 intensity overrides
            needs_mask = (not include_all_cells and (circ_thresh > 0 or round_thresh > 0)) or \
                         (current_option == "ER/PR analysis" and cls_id == 0) or \
                         (current_option in ["KI67 analysis", "ER/PR analysis"] and cls_id == 1)
            
            if needs_mask:
                pad = 2
                rx1, ry1 = max(0, x1 - pad), max(0, y1 - pad)
                rx2, ry2 = min(W, x2 + pad), min(H, y2 + pad)
                
                roi = img_bgr[ry1:ry2, rx1:rx2]
                if roi.size > 0:
                    # Create mask within bounding box to isolate the actual cell shape
                    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    gray_roi = cv2.GaussianBlur(gray_roi, (3, 3), 0)
                    _, mask = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                    mask_valid = True
            
            # Apply strict filtering if Include All Cells is OFF
            if not include_all_cells:
                # 1. Size Filter
                if size_pct_thresh > 0 and box_area < min_area_thresh:
                    continue
                    
                # 2. Shape Filter (Cellularity & Roundness)
                if mask_valid and (circ_thresh > 0 or round_thresh > 0):
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if not contours:
                        skip_cell = True
                    else:
                        largest_contour = max(contours, key=cv2.contourArea)
                        contour_area = cv2.contourArea(largest_contour)
                        perimeter = cv2.arcLength(largest_contour, True)
                        
                        if perimeter > 0:
                            circularity = 4 * np.pi * (contour_area / (perimeter ** 2))
                            if circularity < circ_thresh:
                                skip_cell = True
                        
                        if not skip_cell and round_thresh > 0:
                            if len(largest_contour) >= 5:
                                rect = cv2.fitEllipse(largest_contour)
                                major_axis = max(rect[1][0], rect[1][1])
                                if major_axis > 0:
                                    roundness = 4 * contour_area / (np.pi * (major_axis ** 2))
                                    if roundness < round_thresh:
                                        skip_cell = True
                            else:
                                rect = cv2.minAreaRect(largest_contour)
                                major_axis = max(rect[1][0], rect[1][1])
                                if major_axis > 0:
                                    roundness = 4 * contour_area / (np.pi * (major_axis ** 2))
                                    if roundness < round_thresh:
                                        skip_cell = True
                elif (circ_thresh > 0 or round_thresh > 0) and not mask_valid:
                    # Drop cell if we require shape thresholds but couldn't compute a mask
                    skip_cell = True
                    
            if skip_cell:
                continue
                                
            # --- Color-based Reclassification ---
            
            # 1. KI67 & ER/PR DAB Override: If YOLO thinks it's Negative, but it's very brown
            if current_option in ["KI67 analysis", "ER/PR analysis"] and cls_id == 1:
                pad = 2
                rx1, ry1 = max(0, x1 - pad), max(0, y1 - pad)
                rx2, ry2 = min(W, x2 + pad), min(H, y2 + pad)
                dab_roi = dab_channel[ry1:ry2, rx1:rx2]
                
                if mask_valid and mask.shape == dab_roi.shape and np.any(mask > 0):
                    mean_dab = np.mean(dab_roi[mask > 0])
                else:
                    mean_dab = np.mean(dab_roi) if dab_roi.size > 0 else 0
                    
                if mean_dab >= 0.15:  # Sufficient brown pigment detected
                    cls_id = 0  # Override YOLO and force to Positive
                    
            # 2. ER/PR Intensity Sub-classification
            if current_option == "ER/PR analysis" and cls_id == 0:
                pad = 2
                rx1, ry1 = max(0, x1 - pad), max(0, y1 - pad)
                rx2, ry2 = min(W, x2 + pad), min(H, y2 + pad)
                dab_roi = dab_channel[ry1:ry2, rx1:rx2]
                
                if mask_valid and mask.shape == dab_roi.shape and np.any(mask > 0):
                    mean_dab = np.mean(dab_roi[mask > 0])
                else:
                    mean_dab = np.mean(dab_roi) if dab_roi.size > 0 else 0
                    
                if mean_dab < 0.35:
                    cls_id = 2 # Weak (Yellow)
                elif mean_dab < 0.60:
                    cls_id = 3 # Moderate (Orange)
                else:
                    cls_id = 4 # Strong (Red)
            
            # Append box coordinates for dynamic UI drawing
            boxes_out.append([x1, y1, x2, y2, cls_id])
            
        return drawn_blend_rgb, boxes_out

    except Exception as e:
        print(f"Inference error: {e}")
        drawn_blend_rgb = img_rgb.copy()

    return drawn_blend_rgb, boxes_out


# -----------------
# Zoomable & Editable Image Viewer Widget
# -----------------
class ImageSliderWidget(QWidget):
    # Emit signal when bounding box is manually added/removed/edited
    edits_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.original_pixmap = None
        self.processed_pixmap = None
        self.boxes = []
        self.slider_pos = 0.5  
        self.is_dragging = False
        self.is_panning = False
        
        # Panning & Zoom variables
        self.zoom = 1.0
        self.pan = QPointF(0.0, 0.0)
        self.last_pan_pos = QPointF(0.0, 0.0)
        self.default_box_size = 16  # Shrunk manually added boxes to reduce clutter
        
        self.mode = 'screenshot' 
        self.placeholder_text = "Select an AI model to begin"
        self.setMouseTracking(True)
        self.setMinimumSize(640, 480)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        # Reset View Button overlay
        self.btn_reset_view = QPushButton("Reset View", self)
        self.btn_reset_view.setStyleSheet("""
            QPushButton {
                background-color: rgba(15, 23, 42, 0.7);
                color: white;
                border: 1px solid #475569;
                border-radius: 4px;
                padding: 4px 8px;
            }
            QPushButton:hover {
                background-color: rgba(30, 41, 59, 0.9);
            }
        """)
        self.btn_reset_view.clicked.connect(self.fit_to_view)
        self.btn_reset_view.setVisible(False)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.btn_reset_view.move(self.width() - self.btn_reset_view.width() - 10, 10)

    def set_placeholder_text(self, text):
        self.placeholder_text = text
        self.update()

    def set_images(self, processed_qimg, original_qimg=None, boxes=None):
        if processed_qimg is None:
            self.processed_pixmap = None
            self.original_pixmap = None
            self.boxes = []
            self.btn_reset_view.setVisible(False)
            self.update()
            return

        self.processed_pixmap = QPixmap.fromImage(processed_qimg)
        if original_qimg:
            self.original_pixmap = QPixmap.fromImage(original_qimg)
            self.slider_pos = 0.5 
        else:
            self.original_pixmap = None
            
        self.boxes = boxes if boxes is not None else []
        self.btn_reset_view.setVisible(True)
        self.fit_to_view()
        self.update()

    def fit_to_view(self):
        """Centers the image into the widget bounds with a 5% margin."""
        if not self.processed_pixmap: 
            return
        
        w_w, w_h = self.width(), self.height()
        i_w, i_h = self.processed_pixmap.width(), self.processed_pixmap.height()
        
        if i_w == 0 or i_h == 0: 
            return
            
        zoom_x = w_w / i_w
        zoom_y = w_h / i_h
        self.zoom = min(zoom_x, zoom_y) * 0.95 
        
        pan_x = (w_w - i_w * self.zoom) / 2
        pan_y = (w_h - i_h * self.zoom) / 2
        self.pan = QPointF(pan_x, pan_y)
        self.update()

    def map_to_image(self, widget_pos):
        """Converts cursor position back to the raw image coordinates."""
        img_x = (widget_pos.x() - self.pan.x()) / self.zoom
        img_y = (widget_pos.y() - self.pan.y()) / self.zoom
        return img_x, img_y

    def add_box_at(self, img_x, img_y, cls_id):
        if not self.processed_pixmap: return
        # Prevent boxes completely outside image boundaries
        if not (0 <= img_x <= self.processed_pixmap.width() and 0 <= img_y <= self.processed_pixmap.height()):
            return
            
        s = self.default_box_size
        x1 = max(0, img_x - s/2)
        y1 = max(0, img_y - s/2)
        x2 = min(self.processed_pixmap.width(), img_x + s/2)
        y2 = min(self.processed_pixmap.height(), img_y + s/2)
        self.boxes.append([x1, y1, x2, y2, cls_id])
        self.update()
        self.edits_changed.emit()
        
    def delete_box_at(self, img_x, img_y):
        for i in reversed(range(len(self.boxes))):
            x1, y1, x2, y2, _ = self.boxes[i]
            if x1 <= img_x <= x2 and y1 <= img_y <= y2:
                self.boxes.pop(i)
                self.update()
                self.edits_changed.emit()
                return
                
    def change_box_class_at(self, img_x, img_y, cls_id):
        for i in reversed(range(len(self.boxes))):
            x1, y1, x2, y2, _ = self.boxes[i]
            if x1 <= img_x <= x2 and y1 <= img_y <= y2:
                self.boxes[i][4] = cls_id
                self.update()
                self.edits_changed.emit()
                return

    def wheelEvent(self, event):
        """Mouse wheel zooms in towards cursor."""
        if not self.processed_pixmap: 
            return
            
        zoom_in_factor = 1.15
        zoom_out_factor = 1.0 / zoom_in_factor
        
        old_zoom = self.zoom
        if event.angleDelta().y() > 0:
            self.zoom *= zoom_in_factor
        else:
            self.zoom *= zoom_out_factor
            
        self.zoom = max(0.05, min(self.zoom, 15.0))
        
        mouse_pos = event.position()
        self.pan = mouse_pos - (mouse_pos - self.pan) * (self.zoom / old_zoom)
        self.update()

    def mousePressEvent(self, event):
        if not self.processed_pixmap: 
            return
            
        split_x_widget = int(self.width() * self.slider_pos)
        if abs(event.position().x() - split_x_widget) < 20:
            self.is_dragging = True
            return
            
        if event.button() == Qt.MouseButton.MiddleButton:
            self.is_panning = True
            self.last_pan_pos = event.position()
        elif event.button() == Qt.MouseButton.LeftButton:
            img_x, img_y = self.map_to_image(event.position())
            # For manual addition, we default to 0 (Strong positive or KI67 positive)
            self.add_box_at(img_x, img_y, 0)
        elif event.button() == Qt.MouseButton.RightButton:
            img_x, img_y = self.map_to_image(event.position())
            self.delete_box_at(img_x, img_y)
            
    def mouseDoubleClickEvent(self, event):
        if not self.processed_pixmap: 
            return
        if event.button() == Qt.MouseButton.LeftButton:
            img_x, img_y = self.map_to_image(event.position())
            self.change_box_class_at(img_x, img_y, 1) # Double click transforms to Negative

    def mouseMoveEvent(self, event):
        if not self.processed_pixmap: 
            return
            
        split_x_widget = int(self.width() * self.slider_pos)
        
        if self.is_dragging:
            new_pos = event.position().x() / self.width()
            self.slider_pos = max(0.0, min(1.0, new_pos))
            self.update()
        elif self.is_panning:
            delta = event.position() - self.last_pan_pos
            self.pan += delta
            self.last_pan_pos = event.position()
            self.update()
        else:
            if abs(event.position().x() - split_x_widget) < 20:
                self.setCursor(Qt.CursorShape.SplitHCursor)
            else:
                self.setCursor(Qt.CursorShape.ArrowCursor)

    def mouseReleaseEvent(self, event):
        self.is_dragging = False
        self.is_panning = False

    def paintEvent(self, event):
        painter = QPainter(self)
        rect = self.rect()

        if self.processed_pixmap is None:
            painter.setPen(QColor("#94a3b8"))
            painter.setFont(QFont("Arial", 18))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, self.placeholder_text)
            return

        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        
        # --- DRAW PROCESSED SIDE (Background) ---
        painter.translate(self.pan)
        painter.scale(self.zoom, self.zoom)
        
        painter.drawPixmap(0, 0, self.processed_pixmap)
        
        # --- DRAW BOXES ---
        for box in self.boxes:
            x1, y1, x2, y2, cls_id = box
            
            # Map class ID to UI colors dynamically
            if cls_id == 0 or cls_id == 4:
                color = QColor(255, 0, 0) # Red (Positive / Strong)
            elif cls_id == 1:
                color = QColor(0, 0, 255) # Blue (Negative)
            elif cls_id == 2:
                color = QColor(255, 255, 0) # Yellow (Weak)
            elif cls_id == 3:
                color = QColor(255, 165, 0) # Orange (Moderate)
            else:
                color = QColor(0, 255, 0) # Fallback

            pen = QPen(color, max(1, int(2 / self.zoom))) # Normalize thickness against zoom
            painter.setPen(pen)
            painter.drawRect(QRectF(x1, y1, x2 - x1, y2 - y1))
            
        # --- DRAW ORIGINAL RAW SIDE (Clipped) ---
        if self.original_pixmap is not None:
            painter.resetTransform()
            split_x_widget = rect.width() * self.slider_pos
            
            # Restrict drawing to the left of the slider
            clip_rect = QRectF(0, 0, split_x_widget, rect.height())
            painter.setClipRect(clip_rect)
            
            painter.translate(self.pan)
            painter.scale(self.zoom, self.zoom)
            painter.drawPixmap(0, 0, self.original_pixmap)
            
            # --- DRAW SLIDER HANDLE ---
            painter.resetTransform()
            painter.setClipping(False) # remove clip so handle draws correctly
            
            painter.setPen(QPen(QColor(255, 255, 255), 3))
            painter.drawLine(QPointF(split_x_widget, 0), QPointF(split_x_widget, rect.height()))
            
            handle_rect = QRectF(split_x_widget - 12, rect.height() / 2 - 20, 24, 40)
            painter.setBrush(QColor(255, 255, 255))
            painter.setPen(QPen(QColor(150, 150, 150), 1))
            painter.drawRoundedRect(handle_rect, 4, 4)
            
            painter.setPen(QPen(QColor(100, 100, 100), 2))
            painter.drawLine(QPointF(split_x_widget - 3, rect.height() / 2 - 8), QPointF(split_x_widget - 3, rect.height() / 2 + 8))
            painter.drawLine(QPointF(split_x_widget + 3, rect.height() / 2 - 8), QPointF(split_x_widget + 3, rect.height() / 2 + 8))


# -----------------
# Snipping Tool Widget (Per Screen)
# -----------------
class SnippingWidget(QWidget):
    """
    A transparent full-screen overlay for a SINGLE monitor.
    Captures the specific monitor's content and handles High-DPI scaling.
    """
    snippet_taken = pyqtSignal(QImage)
    closed = pyqtSignal()

    def __init__(self, screen):
        super().__init__()
        self.target_screen = screen
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.Tool)
        
        self.setGeometry(screen.geometry())
        self.original_pixmap = screen.grabWindow(0)
        self.setCursor(Qt.CursorShape.CrossCursor)
        
        self.begin = QPoint()
        self.end = QPoint()
        self.is_snipping = False

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawPixmap(self.rect(), self.original_pixmap)
        painter.fillRect(self.rect(), QColor(0, 0, 0, 100))

        if self.is_snipping and self.begin != self.end:
            rect = QRect(self.begin, self.end).normalized()
            painter.save()
            painter.setClipRect(rect)
            painter.drawPixmap(self.rect(), self.original_pixmap)
            painter.restore()
            
            pen = QPen(QColor(255, 0, 0), 2)
            painter.setPen(pen)
            painter.drawRect(rect)

    def mousePressEvent(self, event):
        self.begin = event.pos()
        self.end = event.pos()
        self.is_snipping = True
        self.update()

    def mouseMoveEvent(self, event):
        self.end = event.pos()
        self.update()

    def mouseReleaseEvent(self, event):
        self.is_snipping = False
        selection_rect = QRect(self.begin, self.end).normalized()
        
        if selection_rect.width() > 10 and selection_rect.height() > 10:
            widget_w = self.width()
            widget_h = self.height()
            img_w = self.original_pixmap.width()
            img_h = self.original_pixmap.height()
            
            scale_x = img_w / widget_w if widget_w > 0 else 1
            scale_y = img_h / widget_h if widget_h > 0 else 1
            
            x = int(selection_rect.x() * scale_x)
            y = int(selection_rect.y() * scale_y)
            w = int(selection_rect.width() * scale_x)
            h = int(selection_rect.height() * scale_y)
            
            crop_rect = QRect(x, y, w, h).intersected(self.original_pixmap.rect())
            captured_img = self.original_pixmap.copy(crop_rect).toImage()
            self.snippet_taken.emit(captured_img)
        
        self.close()
        self.closed.emit()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.close()
            self.closed.emit()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Nuclear stain counter (CPU Only)")
        
        # Add window icon
        icon_path = os.path.join("essential_files", "microscope.ico")
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))
            
        self.setMinimumSize(1200, 750)
        self.setStyleSheet(get_stylesheet())
        
        self.was_maximized = False
        self.active_snippers = []
        self.last_screenshot_rgb = None
        
        self.current_yolo_sahi_model = None
        
        self.init_ui()
        self.load_settings()

        # Set default selection
        default_opt = "KI67 analysis"
        if default_opt in MODEL_PATHS:
            self.combo.setCurrentText(default_opt)

        self._load_screenshot_models()
        self.update_active_model_label()

    def _load_screenshot_models(self):
        opt = self.combo.currentText()
        model_pref = 'Yolo_SAHI'

        self.set_status(f'Loading model: {model_pref} for {opt}...')
        QApplication.processEvents()

        yolo_path = MODEL_PATHS.get(opt)
        self.current_yolo_sahi_model = None

        if HAS_SAHI and yolo_path and os.path.exists(yolo_path):
            try:
                self.current_yolo_sahi_model = AutoDetectionModel.from_pretrained(
                    model_type='ultralytics',
                    model_path=yolo_path,
                    confidence_threshold=CONF_THRESHOLD,
                    device='cpu' # Force CPU
                )
                print(f'Loaded YOLO SAHI model from {yolo_path}')
            except TypeError:
                try:
                    self.current_yolo_sahi_model = AutoDetectionModel.from_pretrained(
                        model_type='yolov8',
                        model_path=yolo_path,
                        confidence_threshold=CONF_THRESHOLD,
                        device='cpu'
                    )
                    print(f'Loaded YOLO SAHI model (yolov8 fallback) from {yolo_path}')
                except Exception as e2:
                    print(f'YOLO SAHI fallback load error: {e2}')
            except Exception as e:
                print(f'YOLO SAHI load error: {e}')

        self.set_status('Status: ready')

    def init_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(15)
        root.setLayout(main_layout)

        # --- Left Panel (Scrollable) ---
        left_scroll = QScrollArea()
        left_scroll.setObjectName("SidePanel")
        left_scroll.setFixedWidth(380)
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        left_inner = QWidget()
        left_inner.setObjectName("SidePanelInner")
        
        left_layout = QVBoxLayout()
        left_layout.setContentsMargins(18, 18, 18, 18)
        left_layout.setSpacing(15)
        left_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        left_inner.setLayout(left_layout)

        # 1. Select Analysis
        self.group_model = QGroupBox("Select analysis")
        layout_model = QVBoxLayout()
        layout_model.setSpacing(10)
        self.combo = QComboBox()
        self.combo.addItems(list(MODEL_PATHS.keys()))
        layout_model.addWidget(self.combo)
        
        self.group_model.setLayout(layout_model)
        left_layout.addWidget(self.group_model)

        # 2. Controls
        self.group_controls = QGroupBox("Controls")
        layout_controls = QVBoxLayout()
        layout_controls.setSpacing(10)
        
        self.btn_take_screenshot = QPushButton("Take screenshot")
        self.btn_take_screenshot.setStyleSheet("background: #0ea5e9; color: white; padding: 12px; font-size: 14px;")
        layout_controls.addWidget(self.btn_take_screenshot)

        settings_layout = QHBoxLayout()
        self.btn_advanced = QPushButton("Advanced settings")
        self.btn_advanced.setCheckable(True)
        self.btn_advanced.setFixedHeight(36)
        self.btn_advanced.setStyleSheet("""
            QPushButton { background: #FFC2BA; color: #1e293b; font-weight: bold; border: 2px solid transparent; }
            QPushButton:checked { background: #ff9e8f; border: 2px solid #ff7a63; color: #0f172a; }
            QPushButton:hover { background: #ffad9f; }
        """)
        
        self.btn_reset = QPushButton("Reset App")
        self.btn_reset.setStyleSheet("background: #64748b; color: white; font-weight: bold; border-radius: 6px;")
        self.btn_reset.setFixedHeight(36)
        
        settings_layout.addWidget(self.btn_advanced, 1)
        settings_layout.addWidget(self.btn_reset, 1)
        layout_controls.addLayout(settings_layout)
        
        # --- Advanced Settings Container ---
        self.advanced_container = QWidget()
        advanced_layout = QVBoxLayout(self.advanced_container)
        advanced_layout.setContentsMargins(0, 5, 0, 0)
        advanced_layout.setSpacing(10)

        def create_adv_row(checkbox_text, slider_min, slider_max, slider_step, slider_val, val_label_text):
            row_w = QWidget()
            row_l = QHBoxLayout(row_w)
            row_l.setContentsMargins(0, 0, 0, 0)
            
            chk = QCheckBox(checkbox_text)
            chk.setStyleSheet("color: #334155; font-weight: 500;")
            
            sld = QSlider(Qt.Orientation.Horizontal)
            sld.setMinimum(slider_min)
            sld.setMaximum(slider_max)
            sld.setSingleStep(slider_step)
            sld.setValue(slider_val)
            sld.setEnabled(False) # Disabled until checked
            
            lbl = QLabel(val_label_text)
            lbl.setStyleSheet("color: #334155; font-weight: bold; min-width: 30px;")
            
            row_l.addWidget(chk)
            row_l.addWidget(sld)
            row_l.addWidget(lbl)
            chk.toggled.connect(sld.setEnabled)
            return row_w, chk, sld, lbl

        # Image Pre-processing Parameters Title
        lbl_preproc = QLabel("<b>Image pre-processing parameters:</b>")
        lbl_preproc.setStyleSheet("color: #334155; font-size: 13px; margin-top: 5px;")
        advanced_layout.addWidget(lbl_preproc)

        # Macenko Checkbox (No slider)
        self.chk_macenko = QCheckBox("Auto-calibrate stain colors (Macenko)")
        self.chk_macenko.setStyleSheet("color: #334155; font-weight: 500;")
        self.chk_macenko.setChecked(True)
        self.chk_macenko.stateChanged.connect(self.on_advanced_setting_changed)
        advanced_layout.addWidget(self.chk_macenko)

        # Stain Normalization (Vahadane)
        vahadane_row, self.chk_vahadane, self.slider_vahadane, self.lbl_vahadane_val = create_adv_row(
            "Stain Norm (Vahadane)", 0, 100, 5, 10, "0.10"
        )
        self.slider_vahadane.valueChanged.connect(lambda v: self.lbl_vahadane_val.setText(f"{v/100.0:.2f}"))
        self.slider_vahadane.sliderReleased.connect(self.on_advanced_setting_changed)
        self.chk_vahadane.stateChanged.connect(self.on_advanced_setting_changed)
        advanced_layout.addWidget(vahadane_row)
        
        # CLAHE
        clahe_row, self.chk_clahe, self.slider_clahe, self.lbl_clahe_val = create_adv_row(
            "CLAHE Enhancement", 1, 100, 5, 5, "0.5"
        )
        self.slider_clahe.valueChanged.connect(lambda v: self.lbl_clahe_val.setText(f"{v/10.0:.1f}"))
        self.slider_clahe.sliderReleased.connect(self.on_advanced_setting_changed)
        self.chk_clahe.stateChanged.connect(self.on_advanced_setting_changed)
        advanced_layout.addWidget(clahe_row)

        # Unsharp Masking
        unsharp_row, self.chk_unsharp, self.slider_unsharp, self.lbl_unsharp_val = create_adv_row(
            "Unsharp Masking", 0, 50, 1, 5, "0.5"
        )
        self.slider_unsharp.valueChanged.connect(lambda v: self.lbl_unsharp_val.setText(f"{v/10.0:.1f}"))
        self.slider_unsharp.sliderReleased.connect(self.on_advanced_setting_changed)
        self.chk_unsharp.stateChanged.connect(self.on_advanced_setting_changed)
        advanced_layout.addWidget(unsharp_row)

        # Cell Filter Parameters Title
        lbl_cell_filters = QLabel("<b>Cell filter parameters:</b>")
        lbl_cell_filters.setStyleSheet("color: #334155; font-size: 13px; margin-top: 10px;")
        advanced_layout.addWidget(lbl_cell_filters)

        # Include All Cells Checkbox
        self.chk_include_all = QCheckBox("Include all cells")
        self.chk_include_all.setStyleSheet("color: #334155; font-weight: 500;")
        self.chk_include_all.setChecked(False)
        self.chk_include_all.stateChanged.connect(self.on_include_all_changed)
        self.chk_include_all.stateChanged.connect(self.on_advanced_setting_changed)
        advanced_layout.addWidget(self.chk_include_all)

        # Advanced Post-processing Filters
        self.size_layout = QHBoxLayout()
        self.lbl_size_title = QLabel("Size ratio (relative to positive):")
        self.lbl_size_title.setStyleSheet("color: #334155; font-weight: 500;")
        self.slider_size = QSlider(Qt.Orientation.Horizontal)
        self.slider_size.setMinimum(0)
        self.slider_size.setMaximum(100)
        self.slider_size.setSingleStep(5)
        self.slider_size.setValue(30)
        self.lbl_size_val = QLabel("30%")
        self.lbl_size_val.setStyleSheet("color: #334155; font-weight: bold; min-width: 30px;")
        self.size_layout.addWidget(self.lbl_size_title)
        self.size_layout.addWidget(self.slider_size)
        self.size_layout.addWidget(self.lbl_size_val)
        advanced_layout.addLayout(self.size_layout)
        
        self.circ_layout = QHBoxLayout()
        self.lbl_circ_title = QLabel("Cellularity:")
        self.lbl_circ_title.setStyleSheet("color: #334155; font-weight: 500;")
        self.slider_circ = QSlider(Qt.Orientation.Horizontal)
        self.slider_circ.setMinimum(0)
        self.slider_circ.setMaximum(100)
        self.slider_circ.setSingleStep(5)
        self.slider_circ.setValue(40)
        self.lbl_circ_val = QLabel("0.40")
        self.lbl_circ_val.setStyleSheet("color: #334155; font-weight: bold; min-width: 30px;")
        self.circ_layout.addWidget(self.lbl_circ_title)
        self.circ_layout.addWidget(self.slider_circ)
        self.circ_layout.addWidget(self.lbl_circ_val)
        advanced_layout.addLayout(self.circ_layout)

        self.round_layout = QHBoxLayout()
        self.lbl_round_title = QLabel("Roundness:")
        self.lbl_round_title.setStyleSheet("color: #334155; font-weight: 500;")
        self.slider_round = QSlider(Qt.Orientation.Horizontal)
        self.slider_round.setMinimum(0)
        self.slider_round.setMaximum(100)
        self.slider_round.setSingleStep(5)
        self.slider_round.setValue(40)
        self.lbl_round_val = QLabel("0.40")
        self.lbl_round_val.setStyleSheet("color: #334155; font-weight: bold; min-width: 30px;")
        self.round_layout.addWidget(self.lbl_round_title)
        self.round_layout.addWidget(self.slider_round)
        self.round_layout.addWidget(self.lbl_round_val)
        advanced_layout.addLayout(self.round_layout)

        # User Toggle for Inverting Classes Dynamically
        self.chk_invert_yolo = QCheckBox("Invert YOLO/SAHI Classes (0=Negative, 1=Positive)")
        self.chk_invert_yolo.setStyleSheet("color: #334155; font-weight: 500; margin-top: 10px;")
        self.chk_invert_yolo.setChecked(False)
        self.chk_invert_yolo.stateChanged.connect(self.on_advanced_setting_changed)
        advanced_layout.addWidget(self.chk_invert_yolo)
        
        # Save / Reset Settings Buttons
        btn_layout = QHBoxLayout()
        self.btn_save_settings = QPushButton("Save Settings")
        self.btn_save_settings.setStyleSheet("background: #10b981; color: white; padding: 6px; font-weight: bold;")
        self.btn_save_settings.clicked.connect(self.save_settings)
        
        self.btn_reset_settings = QPushButton("Reset Settings")
        self.btn_reset_settings.setStyleSheet("background: #f43f5e; color: white; padding: 6px; font-weight: bold;")
        self.btn_reset_settings.clicked.connect(self.on_reset_settings_clicked)
        
        btn_layout.addWidget(self.btn_save_settings)
        btn_layout.addWidget(self.btn_reset_settings)
        
        btn_widget = QWidget()
        btn_widget.setLayout(btn_layout)
        btn_layout.setContentsMargins(0, 10, 0, 0)
        advanced_layout.addWidget(btn_widget)

        self.advanced_container.setVisible(False)
        layout_controls.addWidget(self.advanced_container)
        
        self.group_controls.setLayout(layout_controls)
        left_layout.addWidget(self.group_controls)

        # 3. Results
        self.group_results = QGroupBox("Results")
        layout_results = QVBoxLayout()
        layout_results.setSpacing(10)

        self.lbl_class0 = QLabel("<b>Positive cells:</b> <span style='color: red;'>0</span>")
        self.lbl_class1 = QLabel("<b>Negative cells:</b> <span style='color: blue;'>0</span>")
        self.lbl_er_details = QLabel("")
        
        for lbl in (self.lbl_class0, self.lbl_class1, self.lbl_er_details):
            lbl.setFont(QFont("Consolas", 11))
            lbl.setStyleSheet("color: #334155;")
            layout_results.addWidget(lbl)
            
        self.lbl_er_details.setVisible(False)

        layout_results.addSpacing(5)
        
        self.lbl_kpi = QLabel("KPI: -")
        self.lbl_kpi.setWordWrap(True)
        self.lbl_kpi.setStyleSheet("""
            font-size: 14px; 
            font-weight: bold; 
            color: #065f46; 
            background-color: #d1fae5; 
            padding: 10px; 
            border-radius: 6px;
            border: 1px solid #a7f3d0;
        """)
        layout_results.addWidget(self.lbl_kpi)

        # ---- Note Label ----
        note_text = (
            "<b>Note:</b><br>"
            "- Use the mouse wheel to zoom in and out of the image<br>"
            "- Right click on an annotation to delete<br>"
            "- Left click once to add a <span style='color: red;'>positive</span> annotation<br>"
            "- Left click twice to add a <span style='color: blue;'>negative</span> annotation<br>"
        )
        self.lbl_note = QLabel(note_text)
        self.lbl_note.setWordWrap(True)
        self.lbl_note.setStyleSheet("color: #334155; font-size: 11px; margin-top: 10px;")
        layout_results.addWidget(self.lbl_note)
        
        self.group_results.setLayout(layout_results)
        left_layout.addWidget(self.group_results)

        left_layout.addStretch()

        # Active Model Info
        self.lbl_active_model = QLabel("Active Engine: -")
        self.lbl_active_model.setWordWrap(True)
        self.lbl_active_model.setStyleSheet("color: #475569; font-size: 11px; font-weight: bold;")
        left_layout.addWidget(self.lbl_active_model)

        # Progress Bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedHeight(6)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: none;
                background-color: #e2e8f0;
                border-radius: 3px;
            }
            QProgressBar::chunk {
                background-color: #3b82f6;
                border-radius: 3px;
            }
        """)
        self.progress_bar.setVisible(False)
        left_layout.addWidget(self.progress_bar)

        # Status
        self.lbl_status = QLabel("Status: ready")
        self.lbl_status.setWordWrap(True)
        self.lbl_status.setStyleSheet("color: #94a3b8; font-size: 11px;")
        left_layout.addWidget(self.lbl_status)
        
        # Disclaimer and References
        disclaimer_text = (
            "<i><b style='color: red;'>Disclaimer:</b> This output is AI generated. "
            "Please verify and confirm results.<br><br>"
            "This app was built using Google Gemini Pro and uses YOLO26 and SAHI.<br><br>"
            "<b>References:</b><ul style='margin-top: 2px;'>"
            "<li>https://github.com/ultralytics/ultralytics</li>"
            "<li>https://github.com/obss/sahi</li>"
            "</ul></i>"
        )
        self.lbl_disclaimer = QLabel(disclaimer_text)
        self.lbl_disclaimer.setWordWrap(True)
        self.lbl_disclaimer.setStyleSheet("color: #64748b; font-size: 10px; margin-top: 15px;")
        left_layout.addWidget(self.lbl_disclaimer)

        left_scroll.setWidget(left_inner)
        main_layout.addWidget(left_scroll)

        # --- Right Panel (Video) ---
        right_panel = QFrame()
        right_panel.setObjectName("VideoPanel")
        right_layout = QVBoxLayout()
        right_layout.setContentsMargins(15, 15, 15, 15)
        right_layout.setSpacing(15)
        right_panel.setLayout(right_layout)

        self.video_widget = ImageSliderWidget()
        self.video_widget.setObjectName("VideoScreen")
        
        right_layout.addWidget(self.video_widget)

        main_layout.addWidget(right_panel, stretch=1)
        
        # Wire up signals
        self.combo.currentTextChanged.connect(self.on_option_changed)
        self.btn_take_screenshot.clicked.connect(self.on_take_screenshot_clicked)
        self.btn_reset.clicked.connect(self.on_reset_clicked)
        self.btn_advanced.toggled.connect(self.advanced_container.setVisible)
        self.video_widget.edits_changed.connect(self.on_recalculate_triggered)
        
        self.slider_size.valueChanged.connect(lambda v: self.lbl_size_val.setText(f"{v}%"))
        self.slider_size.sliderReleased.connect(self.on_advanced_setting_changed)

        self.slider_circ.valueChanged.connect(lambda v: self.lbl_circ_val.setText(f"{v/100.0:.2f}"))
        self.slider_circ.sliderReleased.connect(self.on_advanced_setting_changed)

        self.slider_round.valueChanged.connect(lambda v: self.lbl_round_val.setText(f"{v/100.0:.2f}"))
        self.slider_round.sliderReleased.connect(self.on_advanced_setting_changed)
        
        # Enforce exact requested defaults at startup if no JSON overrides them
        self.chk_macenko.setChecked(True)
        self.chk_vahadane.setChecked(True)
        self.slider_vahadane.setValue(10)
        self.slider_vahadane.setEnabled(True)
        self.chk_clahe.setChecked(True)
        self.slider_clahe.setValue(5)
        self.slider_clahe.setEnabled(True)
        self.chk_unsharp.setChecked(True)
        self.slider_unsharp.setValue(5)
        self.slider_unsharp.setEnabled(True)

    def on_include_all_changed(self):
        is_checked = self.chk_include_all.isChecked()
        # Toggle visual enabling of sliders based on 'Include all cells'
        self.slider_size.setEnabled(not is_checked)
        self.lbl_size_title.setEnabled(not is_checked)
        self.lbl_size_val.setEnabled(not is_checked)

        self.slider_circ.setEnabled(not is_checked)
        self.lbl_circ_title.setEnabled(not is_checked)
        self.lbl_circ_val.setEnabled(not is_checked)

        self.slider_round.setEnabled(not is_checked)
        self.lbl_round_title.setEnabled(not is_checked)
        self.lbl_round_val.setEnabled(not is_checked)

    def load_settings(self):
        settings_path = os.path.join("essential_files", "settings.json")
        if os.path.exists(settings_path):
            try:
                with open(settings_path, 'r') as f:
                    data = json.load(f)
                
                # Block signals to prevent processing multiple times during load
                self.chk_invert_yolo.blockSignals(True)
                self.chk_macenko.blockSignals(True)
                self.chk_clahe.blockSignals(True)
                self.chk_unsharp.blockSignals(True)
                self.chk_vahadane.blockSignals(True)
                self.chk_include_all.blockSignals(True)
                
                # Apply states
                self.chk_invert_yolo.setChecked(data.get("invert_classes", False))
                self.chk_macenko.setChecked(data.get("macenko_enabled", True))
                
                self.chk_clahe.setChecked(data.get("clahe_enabled", True))
                self.slider_clahe.setValue(data.get("clahe_value", 5))
                
                self.chk_unsharp.setChecked(data.get("unsharp_enabled", True))
                self.slider_unsharp.setValue(data.get("unsharp_value", 5))
                
                self.chk_vahadane.setChecked(data.get("vahadane_enabled", True))
                self.slider_vahadane.setValue(data.get("vahadane_value", 10))
                
                self.chk_include_all.setChecked(data.get("include_all_cells", False))
                self.slider_size.setValue(data.get("size_value", 30))
                self.slider_circ.setValue(data.get("circ_value", 40))
                self.slider_round.setValue(data.get("round_value", 40))
                
                # Enable sliders if checked
                self.slider_vahadane.setEnabled(self.chk_vahadane.isChecked())
                self.slider_clahe.setEnabled(self.chk_clahe.isChecked())
                self.slider_unsharp.setEnabled(self.chk_unsharp.isChecked())
                
                # Visually update enablement based on 'Include all cells'
                self.on_include_all_changed()
                
                # Update text labels
                self.lbl_vahadane_val.setText(f"{self.slider_vahadane.value()/100.0:.2f}")
                self.lbl_clahe_val.setText(f"{self.slider_clahe.value()/10.0:.1f}")
                self.lbl_unsharp_val.setText(f"{self.slider_unsharp.value()/10.0:.1f}")
                self.lbl_size_val.setText(f"{self.slider_size.value()}%")
                self.lbl_circ_val.setText(f"{self.slider_circ.value()/100.0:.2f}")
                self.lbl_round_val.setText(f"{self.slider_round.value()/100.0:.2f}")

                # Unblock signals
                self.chk_invert_yolo.blockSignals(False)
                self.chk_macenko.blockSignals(False)
                self.chk_clahe.blockSignals(False)
                self.chk_unsharp.blockSignals(False)
                self.chk_vahadane.blockSignals(False)
                self.chk_include_all.blockSignals(False)
                
            except Exception as e:
                self.set_status(f"Error loading settings: {e}")

    def save_settings(self):
        os.makedirs("essential_files", exist_ok=True)
        settings_path = os.path.join("essential_files", "settings.json")
        data = {
            "invert_classes": self.chk_invert_yolo.isChecked(),
            "macenko_enabled": self.chk_macenko.isChecked(),
            "clahe_enabled": self.chk_clahe.isChecked(),
            "clahe_value": self.slider_clahe.value(),
            "unsharp_enabled": self.chk_unsharp.isChecked(),
            "unsharp_value": self.slider_unsharp.value(),
            "vahadane_enabled": self.chk_vahadane.isChecked(),
            "vahadane_value": self.slider_vahadane.value(),
            "include_all_cells": self.chk_include_all.isChecked(),
            "size_value": self.slider_size.value(),
            "circ_value": self.slider_circ.value(),
            "round_value": self.slider_round.value()
        }
        try:
            with open(settings_path, 'w') as f:
                json.dump(data, f, indent=4)
            self.set_status("Settings saved successfully.")
        except Exception as e:
            self.set_status(f"Error saving settings: {e}")

    def on_reset_settings_clicked(self):
        # Block signals to prevent continuous reprocessing
        self.chk_invert_yolo.blockSignals(True)
        self.chk_macenko.blockSignals(True)
        self.chk_clahe.blockSignals(True)
        self.chk_unsharp.blockSignals(True)
        self.chk_vahadane.blockSignals(True)
        self.chk_include_all.blockSignals(True)
        
        # Restore Requested Defaults
        self.chk_invert_yolo.setChecked(False)
        self.chk_macenko.setChecked(True)
        self.chk_clahe.setChecked(True)
        self.slider_clahe.setValue(5)
        self.chk_unsharp.setChecked(True)
        self.slider_unsharp.setValue(5)
        self.chk_vahadane.setChecked(True)
        self.slider_vahadane.setValue(10)
        self.chk_include_all.setChecked(False)
        
        self.slider_size.setValue(30)
        self.slider_circ.setValue(40)
        self.slider_round.setValue(40)
        
        # Ensure enabled states
        self.slider_clahe.setEnabled(True)
        self.slider_unsharp.setEnabled(True)
        self.slider_vahadane.setEnabled(True)
        self.on_include_all_changed()
        
        # Update text labels
        self.lbl_clahe_val.setText("0.5")
        self.lbl_unsharp_val.setText("0.5")
        self.lbl_vahadane_val.setText("0.10")
        self.lbl_size_val.setText("30%")
        self.lbl_circ_val.setText("0.40")
        self.lbl_round_val.setText("0.40")
        
        # Unblock signals
        self.chk_invert_yolo.blockSignals(False)
        self.chk_macenko.blockSignals(False)
        self.chk_clahe.blockSignals(False)
        self.chk_unsharp.blockSignals(False)
        self.chk_vahadane.blockSignals(False)
        self.chk_include_all.blockSignals(False)
        
        self.save_settings()
        self.on_advanced_setting_changed()

    def update_active_model_label(self):
        if getattr(self, 'current_yolo_sahi_model', None):
            self.lbl_active_model.setText("Active Engine: YOLO SAHI (CPU Mode)")
        else:
            self.lbl_active_model.setText("Active Engine: None detected (Error/Loading)")

    def on_advanced_setting_changed(self):
        if self.last_screenshot_rgb is not None:
            self.set_status("Reprocessing captured region with new settings...")
            self.process_screenshot()

    def on_recalculate_triggered(self):
        """Forces KPI and count recalculation dynamically on manual UI edits"""
        if self.last_screenshot_rgb is None or not hasattr(self.video_widget, 'boxes'):
            return

        current_opt = self.combo.currentText()
        boxes = self.video_widget.boxes
        
        counts = {'pos': 0, 'neg': 0, 'weak': 0, 'mod': 0, 'strong': 0, 'total': 0}
        for b in boxes:
            cid = b[4]
            if cid == 1:
                counts['neg'] += 1
            elif current_opt == "ER/PR analysis":
                if cid == 2: counts['weak'] += 1
                elif cid == 3: counts['mod'] += 1
                elif cid in (0, 4): counts['strong'] += 1
            else:
                if cid == 0: counts['pos'] += 1
                
        if current_opt == "ER/PR analysis":
            counts['pos'] = counts['weak'] + counts['mod'] + counts['strong']
            counts['total'] = counts['pos'] + counts['neg']
            c_pos = counts['pos']
            total = counts['total']
            
            pct = (c_pos / total * 100.0) if total > 0 else 0.0
            
            ps = 0
            if total > 0:
                prop = c_pos / total
                if prop == 0: ps = 0
                elif prop < 0.01: ps = 1
                elif prop <= 0.10: ps = 2
                elif prop <= 0.33: ps = 3
                elif prop <= 0.66: ps = 4
                else: ps = 5
                
            is_score = 0
            if c_pos > 0:
                is_score = round((counts['weak']*1 + counts['mod']*2 + counts['strong']*3) / c_pos)
                
            allred = ps + is_score
            status = "Positive" if allred >= 3 else "Negative"
            
            kpi_text = (f"ER/PR positive cells: {pct:.1f}%<br><br>"
                        f"Allred score: {allred}/8 ({status})<br>"
                        f"<span style='font-weight: normal;'>Proportion: {ps}/5, Intensity: {is_score}/3</span>")
            
            payload = {
                'class_counts': {
                    'class_pos': c_pos, 'class_neg': counts['neg'],
                    'class_weak': counts['weak'], 'class_mod': counts['mod'],
                    'class_strong': counts['strong'], 'class_total': total
                },
                'kpi_text': kpi_text
            }
        else:
            c_pos = counts['pos']
            c_neg = counts['neg']
            denom = c_pos + c_neg
            pct = (c_pos / denom * 100.0) if denom > 0 else 0.0
            
            kpi_text = f"KI67 index : {pct:.1f}%"
            payload = {
                'class_counts': {'class_0': c_pos, 'class_1': c_neg},
                'kpi_text': kpi_text
            }
            
        self.update_counts(payload)
        self.set_status("Cell counts recalculated.")

    def on_reset_clicked(self):
        self.last_screenshot_rgb = None
        self.video_widget.set_placeholder_text("Select an AI model to begin")
        self.update_frame(None)
        names = CLASS_NAMES.get(self.combo.currentText(), ["Positive cells", "Negative cells"])
        self.lbl_class0.setText(f"<b>{names[0]}:</b> <span style='color: red;'>0</span>")
        self.lbl_class1.setText(f"<b>{names[1]}:</b> <span style='color: blue;'>0</span>")
        self.lbl_er_details.setVisible(False)
        self.lbl_kpi.setText("KPI: -")
        self.set_status("Results cleared.")
            
    def on_take_screenshot_clicked(self):
        self.update_active_model_label()
        self.advanced_container.setVisible(self.btn_advanced.isChecked())
        self.on_start_capture()

    def on_start_capture(self):
        self.set_status("Select region on any monitor...")
        self.close_all_snippers()
        
        self.was_maximized = self.isMaximized()
        self.showMinimized()

        screens = QGuiApplication.screens()
        for screen in screens:
            snipper = SnippingWidget(screen)
            snipper.snippet_taken.connect(self.on_image_captured)
            snipper.closed.connect(self.on_snipper_closed)
            snipper.show()
            self.active_snippers.append(snipper)

    def close_all_snippers(self):
        for s in self.active_snippers:
            s.close()
        self.active_snippers.clear()
        
    def restore_main_window(self):
        if self.was_maximized:
            self.showMaximized()
        else:
            self.showNormal()
        self.activateWindow()
        self.raise_()
        QApplication.processEvents()
        
    def on_snipper_closed(self):
        if self.isMinimized():
            self.restore_main_window()

    def on_image_captured(self, qimg: QImage):
        self.close_all_snippers()
        self.set_status("Processing captured region...")
        self.restore_main_window()
        
        # Show processing text visually while the user waits
        self.video_widget.set_placeholder_text("Processing image...")
        self.video_widget.set_images(None)
        QApplication.processEvents()

        qimg = qimg.convertToFormat(QImage.Format.Format_RGB888)
        w, h = qimg.width(), qimg.height()
        
        ptr = qimg.bits()
        ptr.setsize(qimg.sizeInBytes())
        
        arr = np.frombuffer(ptr, np.uint8).reshape(h, qimg.bytesPerLine())
        arr = arr[:, :w*3].reshape(h, w, 3)
        self.last_screenshot_rgb = arr.copy() # Store the pure RGB format
        
        self.process_screenshot()

    def process_screenshot(self):
        if self.last_screenshot_rgb is None:
            return
            
        frame_rgb = self.last_screenshot_rgb
        
        current_opt = self.combo.currentText()
        invert_classes = self.chk_invert_yolo.isChecked()
        
        # Capture pre-processing settings
        use_macenko = self.chk_macenko.isChecked()
        
        use_vahadane = self.chk_vahadane.isChecked()
        vahadane_val = self.slider_vahadane.value() / 100.0
        
        use_clahe = self.chk_clahe.isChecked()
        clahe_val = self.slider_clahe.value() / 10.0
        
        use_unsharp = self.chk_unsharp.isChecked()
        unsharp_val = self.slider_unsharp.value() / 10.0
        
        # Capture shape filtering settings
        include_all_cells = self.chk_include_all.isChecked()
        size_pct_thresh = self.slider_size.value() / 100.0
        circ_thresh = self.slider_circ.value() / 100.0
        round_thresh = self.slider_round.value() / 100.0
        
        model_to_use = None
        if getattr(self, 'current_yolo_sahi_model', None):
            model_to_use = self.current_yolo_sahi_model

        if model_to_use is None:
            self.set_status("Error: Appropriate model for screenshot analysis not found or failed to load.")
            self.video_widget.set_placeholder_text("Error: Model not loaded")
            self.video_widget.update()
            self.btn_take_screenshot.setEnabled(True)
            self.btn_reset.setEnabled(True)
            return

        self.update_active_model_label()

        def update_progress(pct, msg):
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(pct)
            self.set_status(f"{pct}% done. {msg}")
            QApplication.processEvents()

        self.btn_take_screenshot.setEnabled(False)

        # Run Universal CPU Inference with Enhancement Options
        drawn_rgb_or_bgr, boxes_out = run_screenshot_inference(
            frame_rgb, model_to_use, current_opt,
            progress_callback=update_progress, invert_classes=invert_classes,
            use_macenko=use_macenko,
            use_vahadane=use_vahadane, vahadane_val=vahadane_val,
            use_clahe=use_clahe, clahe_val=clahe_val,
            use_unsharp=use_unsharp, unsharp_val=unsharp_val,
            include_all_cells=include_all_cells,
            size_pct_thresh=size_pct_thresh, circ_thresh=circ_thresh, round_thresh=round_thresh
        )
        
        self.progress_bar.setVisible(False)
        self.progress_bar.setValue(0)
        
        h, w, _ = frame_rgb.shape
        bpl = 3 * w
        
        # Original Image (always RGB raw capture)
        qt_orig = QImage(frame_rgb.data, w, h, bpl, QImage.Format.Format_RGB888).copy()

        # The base Enhanced Image (without hardcoded boxes)
        qt_res = QImage(drawn_rgb_or_bgr.data, w, h, bpl, QImage.Format.Format_RGB888).copy()
        
        # Set images AND vector box coordinates directly to the UI Widget
        self.video_widget.set_images(qt_res, original_qimg=qt_orig, boxes=boxes_out)
        
        # Trigger KPI calculation based on exactly what was injected into the viewer
        self.on_recalculate_triggered()
        
        self.set_status("Capture analysis complete. Hover to pan/zoom, click to edit boxes.")
        self.btn_take_screenshot.setEnabled(True)

    def update_frame(self, qimage: QImage, orig_qimage: QImage = None):
        self.video_widget.set_images(qimage, orig_qimage)

    def update_counts(self, payload: dict):
        cn = payload.get('class_counts', {})
        
        if self.combo.currentText() == "ER/PR analysis":
            p = cn.get("class_pos", 0)
            n = cn.get("class_neg", 0)
            w = cn.get("class_weak", 0)
            m = cn.get("class_mod", 0)
            s = cn.get("class_strong", 0)
            
            self.lbl_class0.setText(f"<b>Positive cells:</b> <span style='color: red;'>{p}</span>")
            self.lbl_class1.setText(f"<b>Negative cells:</b> <span style='color: blue;'>{n}</span>")
            
            self.lbl_er_details.setText(
                f"<b>Intensity breakdown:</b><br>"
                f"Weak cells: <span style='color: #b8b800;'>{w}</span><br>"
                f"Moderate cells: <span style='color: #ffa500;'>{m}</span><br>"
                f"Strong cells: <span style='color: red;'>{s}</span>"
            )
            self.lbl_er_details.setVisible(True)
        else:
            c0 = cn.get("class_0", 0)
            c1 = cn.get("class_1", 0)
            self.lbl_class0.setText(f"<b>Positive cells:</b> <span style='color: red;'>{c0}</span>")
            self.lbl_class1.setText(f"<b>Negative cells:</b> <span style='color: blue;'>{c1}</span>")
            self.lbl_er_details.setVisible(False)

        self.lbl_kpi.setText(payload.get('kpi_text', ''))

    def set_status(self, text: str):
        self.lbl_status.setText(f"Status: {text}")

    def on_option_changed(self, option_text: str):
        try:
            self.lbl_class0.setText(f"<b>Positive cells:</b> <span style='color: red;'>0</span>")
            self.lbl_class1.setText(f"<b>Negative cells:</b> <span style='color: blue;'>0</span>")
            self.lbl_kpi.setText("KPI: -")
            self.lbl_er_details.setVisible(False)
            
            self._load_screenshot_models()
            self.update_active_model_label()
            
            if self.last_screenshot_rgb is not None:
                self.process_screenshot()
                
        except Exception as e:
            self.set_status(str(e))

def main():
    if hasattr(Qt.HighDpiScaleFactorRoundingPolicy, 'PassThrough'):
        QGuiApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
        
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()