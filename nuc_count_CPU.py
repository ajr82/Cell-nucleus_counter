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

# Try to import ultralytics YOLO
try:
    from ultralytics import YOLO
except Exception as e:
    YOLO = None
    _ultralytics_import_error = e

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
IOU_THRESHOLD = 0.45

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
def apply_clahe(img_bgr, clip_limit=2.0):
    """Applies Contrast Limited Adaptive Histogram Equalization to the L channel in LAB color space."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=max(0.1, float(clip_limit)), tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

def apply_unsharp_mask(img_bgr, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
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

def apply_gamma(img_bgr, gamma=1.0):
    """Applies Gamma Correction to the image."""
    inv_gamma = 1.0 / (gamma + 1e-6)
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(img_bgr, table)

def apply_median_blur(img_bgr, ksize=3):
    """Applies Median Blur to reduce noise while preserving edges."""
    ksize = max(1, int(ksize))
    if ksize % 2 == 0: 
        ksize += 1
    return cv2.medianBlur(img_bgr, ksize)

def apply_reinhard(img_bgr, alpha=1.0):
    """Applies Reinhard stain normalization using stable standard target statistics."""
    # Typical target statistics for a well-balanced DAB/Hematoxylin stain in LAB space
    target_means = np.array([140.0, 135.0, 130.0]) 
    target_stds = np.array([35.0, 10.0, 15.0])
    
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    means, stds = cv2.meanStdDev(lab)
    means = np.array(means).flatten()
    stds = np.array(stds).flatten()
    
    stds = np.maximum(stds, 1e-5) # Prevent division by zero
    
    for i in range(3):
        lab[:, :, i] = ((lab[:, :, i] - means[i]) * (target_stds[i] / stds[i])) + target_means[i]
        
    lab = np.clip(lab, 0, 255).astype(np.uint8)
    norm_bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    # Blend image based on slider strength
    if alpha >= 1.0:
        return norm_bgr
    return cv2.addWeighted(norm_bgr, alpha, img_bgr, 1.0 - alpha, 0)

def apply_vahadane(img_bgr, alpha=1.0):
    """Applies Vahadane-style structure-preserving stain normalization."""
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

def apply_bilateral(img_bgr, d=9):
    """Applies Bilateral Filter to reduce noise while keeping edges sharp."""
    return cv2.bilateralFilter(img_bgr, max(1, int(d)), 75, 75)

def apply_sobel(img_bgr, alpha=0.5):
    """Applies Sobel Edge Filter blended with the original image."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3)
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    sobel_edges = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    sobel_bgr = cv2.cvtColor(sobel_edges, cv2.COLOR_GRAY2BGR)
    if alpha >= 1.0: return sobel_bgr
    return cv2.addWeighted(sobel_bgr, alpha, img_bgr, 1.0 - alpha, 0)

def apply_canny(img_bgr, alpha=0.5):
    """Applies Canny Edge Filter blended with the original image."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    if alpha >= 1.0: return edges_bgr
    return cv2.addWeighted(edges_bgr, alpha, img_bgr, 1.0 - alpha, 0)


# -----------------
# Helper Function for Universal Inference (Screenshot Mode)
# -----------------
def run_screenshot_inference(img_rgb, model, current_option, class_names_map, 
                           progress_callback=None, invert_classes=False,
                           use_clahe=False, clahe_val=2.0,
                           use_unsharp=False, unsharp_val=1.0,
                           use_gamma=False, gamma_val=1.0,
                           use_blur=False, blur_val=3,
                           use_bilateral=False, bilateral_val=9,
                           use_norm=False, norm_val=1.0,
                           use_vahadane=False, vahadane_val=1.0,
                           use_sobel=False, sobel_val=0.5,
                           use_canny=False, canny_val=0.5):
    """
    Universal wrapper to handle optimized CPU SAHI/YOLO.
    """
    counts = {0: 0, 1: 0} # Default fallback counts
    kpi_text = ""
    active_class_names = class_names_map.get(current_option, ["Positive cells", "Negative cells"])
    boxes_out = []
    
    # 0 = Positive, 1 = Negative. Toggle cleanly inverts this mapping.
    pos_class_id = 1 if invert_classes else 0
    neg_class_id = 0 if invert_classes else 1
    
    try:
        if progress_callback:
            progress_callback(5, "Running YOLO_SAHI network on CPU...")
            
        # Conversion ensuring stains are correctly colored for the YOLO model processing
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        
        # --- Apply pre-processing enhancements ---
        if use_blur:
            if progress_callback:
                progress_callback(6, "Applying Median Blur...")
            img_bgr = apply_median_blur(img_bgr, blur_val)

        if use_bilateral:
            if progress_callback:
                progress_callback(8, "Applying Bilateral Filtering...")
            img_bgr = apply_bilateral(img_bgr, bilateral_val)
            
        if use_norm:
            if progress_callback:
                progress_callback(10, "Applying Reinhard Normalization...")
            img_bgr = apply_reinhard(img_bgr, norm_val)
            
        if use_vahadane:
            if progress_callback:
                progress_callback(12, "Applying Vahadane Normalization...")
            img_bgr = apply_vahadane(img_bgr, vahadane_val)
            
        if use_gamma:
            if progress_callback:
                progress_callback(14, "Applying Gamma Correction...")
            img_bgr = apply_gamma(img_bgr, gamma_val)

        if use_clahe:
            if progress_callback:
                progress_callback(16, "Applying CLAHE enhancement...")
            img_bgr = apply_clahe(img_bgr, clahe_val)
            
        if use_unsharp:
            if progress_callback:
                progress_callback(18, "Applying Unsharp Masking...")
            img_bgr = apply_unsharp_mask(img_bgr, amount=unsharp_val)
            
        if use_sobel:
            if progress_callback:
                progress_callback(20, "Applying Sobel Edge Filter...")
            img_bgr = apply_sobel(img_bgr, sobel_val)
            
        if use_canny:
            if progress_callback:
                progress_callback(22, "Applying Canny Edge Filter...")
            img_bgr = apply_canny(img_bgr, canny_val)
            
        # The returned image will be solely the processed RGB (no boxes drawn natively)
        drawn_blend_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # --- EVALUATE AS DETECTION BOUNDING BOXES ONLY ---
        results = get_sliced_prediction(
            image=img_bgr, # SAHI slices the BGR image directly so YOLO sees proper colors
            detection_model=model,
            slice_height=640,
            slice_width=640,
            overlap_height_ratio=0.15,
            overlap_width_ratio=0.15,
            verbose=False
        )
        
        for obj in results.object_prediction_list:
            # Stable bounding box extraction across SAHI versions
            x1 = int(obj.bbox.minx)
            y1 = int(obj.bbox.miny)
            x2 = int(obj.bbox.maxx)
            y2 = int(obj.bbox.maxy)
            
            raw_cls_id = int(obj.category.id)
            
            # --- STRICT SAHI INVERSION MAPPING ---
            # SAHI inherently assigns swapped indices (alphabetical override)
            # We explicitly swap 0 to 1, and 1 to 0 here to normalize it.
            cls_id = 1 - raw_cls_id
            
            is_pos = False
            is_neg = False
            
            if cls_id == pos_class_id:
                is_pos = True
            elif cls_id == neg_class_id:
                is_neg = True
            
            if is_pos:
                counts[0] = counts.get(0, 0) + 1  # 0 is the UI 'Positive' slot
            elif is_neg:
                counts[1] = counts.get(1, 0) + 1  # 1 is the UI 'Negative' slot
            else:
                counts[cls_id] = counts.get(cls_id, 0) + 1
                
            # Append box coordinates for dynamic UI drawing
            # List format: [x1, y1, x2, y2, cls_id]
            boxes_out.append([x1, y1, x2, y2, cls_id])
            
        # --- DETECTION KPI CALCULATION ---
        c0 = counts.get(0, 0)
        c1 = counts.get(1, 0)
        denom = c0 + c1
        pct = (c0 / denom * 100.0) if denom > 0 else 0.0
        
        if current_option == "ER/PR analysis":
            kpi_text = f"ER/PR Positive Index: {pct:.1f}%"
        else:
            kpi_text = f"KI67 index : {pct:.1f}%"
            
        return drawn_blend_rgb, counts, kpi_text, active_class_names, boxes_out

    except Exception as e:
        print(f"Inference error: {e}")
        # Return original RGB if there's an error
        drawn_blend_rgb = img_rgb.copy()
        kpi_text = "Inference Error"

    return drawn_blend_rgb, counts, kpi_text, active_class_names, boxes_out


# -----------------
# Zoomable & Editable Image Viewer Widget
# -----------------
class ImageSliderWidget(QWidget):
    # Add a signal to emit when an edit occurs
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
        self.default_box_size = 20  # Reduced by 60% for manually added boxes
        
        self.mode = 'camera' 
        self.placeholder_text = "Select an AI model to begin"
        self.setMouseTracking(True)
        self.setMinimumSize(640, 480)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        # Reset View Button
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
        # Reposition the reset button to the top right corner
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
            self.mode = 'screenshot'
            self.slider_pos = 0.5 
        else:
            self.original_pixmap = None
            
        self.boxes = boxes if boxes is not None else []
        self.btn_reset_view.setVisible(True)
        self.fit_to_view()
        self.update()

    def fit_to_view(self):
        """Calculates initial pan and zoom to fit the image into the view with 5% margin."""
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
        """Converts widget QPointF to underlying Image coordinates."""
        img_x = (widget_pos.x() - self.pan.x()) / self.zoom
        img_y = (widget_pos.y() - self.pan.y()) / self.zoom
        return img_x, img_y

    def add_box_at(self, img_x, img_y, cls_id):
        """Manually inserts a new bounding box at the clicked coordinates."""
        if not self.processed_pixmap: return
        
        # Boundary check: ensure the click is actually inside the image
        if not (0 <= img_x <= self.processed_pixmap.width() and 0 <= img_y <= self.processed_pixmap.height()):
            return
            
        s = self.default_box_size
        x1 = max(0, img_x - s/2)
        y1 = max(0, img_y - s/2)
        x2 = min(self.processed_pixmap.width(), img_x + s/2)
        y2 = min(self.processed_pixmap.height(), img_y + s/2)
        self.boxes.append([x1, y1, x2, y2, cls_id])
        self.update()
        self.edits_changed.emit() # Auto-trigger recalculation
        
    def delete_box_at(self, img_x, img_y):
        """Deletes the bounding box found under the click."""
        for i in reversed(range(len(self.boxes))):
            x1, y1, x2, y2, _ = self.boxes[i]
            if x1 <= img_x <= x2 and y1 <= img_y <= y2:
                self.boxes.pop(i)
                self.update()
                self.edits_changed.emit() # Auto-trigger recalculation
                return
                
    def change_box_class_at(self, img_x, img_y, cls_id):
        """Changes the class of the bounding box located at the click."""
        for i in reversed(range(len(self.boxes))):
            x1, y1, x2, y2, _ = self.boxes[i]
            if x1 <= img_x <= x2 and y1 <= img_y <= y2:
                self.boxes[i][4] = cls_id
                self.update()
                self.edits_changed.emit() # Auto-trigger recalculation
                return

    def wheelEvent(self, event):
        """Handles zooming with the mouse wheel toward the cursor position."""
        if not self.processed_pixmap: 
            return
            
        zoom_in_factor = 1.15
        zoom_out_factor = 1.0 / zoom_in_factor
        
        old_zoom = self.zoom
        if event.angleDelta().y() > 0:
            self.zoom *= zoom_in_factor
        else:
            self.zoom *= zoom_out_factor
            
        self.zoom = max(0.1, min(self.zoom, 15.0))
        
        mouse_pos = event.position()
        self.pan = mouse_pos - (mouse_pos - self.pan) * (self.zoom / old_zoom)
        self.update()

    def mousePressEvent(self, event):
        if self.mode != 'screenshot' or not self.processed_pixmap: 
            return
            
        split_x_widget = int(self.width() * self.slider_pos)
        
        # Priority 1: Check Slider interaction
        if abs(event.position().x() - split_x_widget) < 20:
            self.is_dragging = True
            return
            
        # Priority 2: Panning/Adding/Deleting
        if event.button() == Qt.MouseButton.MiddleButton:
            self.is_panning = True
            self.last_pan_pos = event.position()
        elif event.button() == Qt.MouseButton.LeftButton:
            img_x, img_y = self.map_to_image(event.position())
            self.add_box_at(img_x, img_y, 0) # 0 = Positive
        elif event.button() == Qt.MouseButton.RightButton:
            img_x, img_y = self.map_to_image(event.position())
            self.delete_box_at(img_x, img_y)
            
    def mouseDoubleClickEvent(self, event):
        if self.mode != 'screenshot' or not self.processed_pixmap: 
            return
            
        if event.button() == Qt.MouseButton.LeftButton:
            img_x, img_y = self.map_to_image(event.position())
            self.change_box_class_at(img_x, img_y, 1) # Change to Negative

    def mouseMoveEvent(self, event):
        if self.mode != 'screenshot' or not self.processed_pixmap: 
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
            color = QColor(255, 0, 0) if cls_id == 0 else QColor(6, 148, 148)
            pen = QPen(color, max(1, int(2 / self.zoom))) # Normalize thickness against zoom
            painter.setPen(pen)
            painter.drawRect(QRectF(x1, y1, x2 - x1, y2 - y1))
            
        # --- DRAW ORIGINAL RAW SIDE (Clipped) ---
        if self.mode == 'screenshot' and self.original_pixmap is not None:
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
        self.setWindowTitle("Nuclear stain counter (CPU)")
        
        # Add window icon
        icon_path = os.path.join("essential_files", "microscope.ico")
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))
            
        self.setMinimumSize(1200, 750)
        self.setStyleSheet(get_stylesheet())
        
        self.was_maximized = False
        self.active_snippers = []
        self.last_screenshot_rgb = None
        self.screenshot_engine_str = "Checking..."
        
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

        if model_pref == 'Yolo_SAHI' and HAS_SAHI and yolo_path and os.path.exists(yolo_path):
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
        
        model_btn_layout = QHBoxLayout()
        self.btn_load = QPushButton("Reload")
        model_btn_layout.addWidget(self.btn_load)
        layout_model.addLayout(model_btn_layout)
        
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
        
        self.btn_reset = QPushButton("Reset")
        self.btn_reset.setStyleSheet("background: #64748b; color: white;")
        self.btn_reset.setFixedHeight(36)
        
        settings_layout.addWidget(self.btn_advanced)
        settings_layout.addWidget(self.btn_reset)
        layout_controls.addLayout(settings_layout)
        
        # --- Advanced Settings Container ---
        self.advanced_container = QWidget()
        advanced_layout = QVBoxLayout(self.advanced_container)
        advanced_layout.setContentsMargins(0, 5, 0, 0)
        advanced_layout.setSpacing(10)

        # Helper method for UI slider + checkbox pairs
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
            sld.setEnabled(False) # Default disabled until checked
            
            lbl = QLabel(val_label_text)
            lbl.setStyleSheet("color: #334155; font-weight: bold; min-width: 30px;")
            
            row_l.addWidget(chk)
            row_l.addWidget(sld)
            row_l.addWidget(lbl)
            
            # Enable slider when checked
            chk.toggled.connect(sld.setEnabled)
            
            return row_w, chk, sld, lbl

        # Gamma Correction
        gamma_row, self.chk_gamma, self.slider_gamma, self.lbl_gamma_val = create_adv_row(
            "Gamma Correction", 1, 30, 1, 10, "1.0"
        )
        self.slider_gamma.valueChanged.connect(lambda v: self.lbl_gamma_val.setText(f"{v/10.0:.1f}"))
        self.slider_gamma.sliderReleased.connect(self.on_advanced_setting_changed)
        self.chk_gamma.stateChanged.connect(self.on_advanced_setting_changed)
        advanced_layout.addWidget(gamma_row)

        # Median Blur
        blur_row, self.chk_blur, self.slider_blur, self.lbl_blur_val = create_adv_row(
            "Median Blur (Noise)", 1, 15, 2, 3, "3"
        )
        self.slider_blur.valueChanged.connect(lambda v: self.lbl_blur_val.setText(f"{v if v%2!=0 else v+1}"))
        self.slider_blur.sliderReleased.connect(self.on_advanced_setting_changed)
        self.chk_blur.stateChanged.connect(self.on_advanced_setting_changed)
        advanced_layout.addWidget(blur_row)

        # Bilateral Filtering
        bilateral_row, self.chk_bilateral, self.slider_bilateral, self.lbl_bilateral_val = create_adv_row(
            "Bilateral Filter", 1, 25, 2, 9, "9"
        )
        self.slider_bilateral.valueChanged.connect(lambda v: self.lbl_bilateral_val.setText(f"{v}"))
        self.slider_bilateral.sliderReleased.connect(self.on_advanced_setting_changed)
        self.chk_bilateral.stateChanged.connect(self.on_advanced_setting_changed)
        advanced_layout.addWidget(bilateral_row)

        # Stain Normalization (Reinhard)
        norm_row, self.chk_norm, self.slider_norm, self.lbl_norm_val = create_adv_row(
            "Stain Norm (Reinhard)", 0, 100, 5, 100, "1.00"
        )
        self.slider_norm.valueChanged.connect(lambda v: self.lbl_norm_val.setText(f"{v/100.0:.2f}"))
        self.slider_norm.sliderReleased.connect(self.on_advanced_setting_changed)
        self.chk_norm.stateChanged.connect(self.on_advanced_setting_changed)
        advanced_layout.addWidget(norm_row)

        # Stain Normalization (Vahadane)
        vahadane_row, self.chk_vahadane, self.slider_vahadane, self.lbl_vahadane_val = create_adv_row(
            "Stain Norm (Vahadane)", 0, 100, 5, 100, "1.00"
        )
        self.slider_vahadane.valueChanged.connect(lambda v: self.lbl_vahadane_val.setText(f"{v/100.0:.2f}"))
        self.slider_vahadane.sliderReleased.connect(self.on_advanced_setting_changed)
        self.chk_vahadane.stateChanged.connect(self.on_advanced_setting_changed)
        advanced_layout.addWidget(vahadane_row)
        
        # CLAHE
        clahe_row, self.chk_clahe, self.slider_clahe, self.lbl_clahe_val = create_adv_row(
            "CLAHE Enhancement", 1, 100, 5, 20, "2.0"
        )
        self.slider_clahe.valueChanged.connect(lambda v: self.lbl_clahe_val.setText(f"{v/10.0:.1f}"))
        self.slider_clahe.sliderReleased.connect(self.on_advanced_setting_changed)
        self.chk_clahe.stateChanged.connect(self.on_advanced_setting_changed)
        advanced_layout.addWidget(clahe_row)

        # Unsharp Masking
        unsharp_row, self.chk_unsharp, self.slider_unsharp, self.lbl_unsharp_val = create_adv_row(
            "Unsharp Masking", 0, 50, 1, 10, "1.0"
        )
        self.slider_unsharp.valueChanged.connect(lambda v: self.lbl_unsharp_val.setText(f"{v/10.0:.1f}"))
        self.slider_unsharp.sliderReleased.connect(self.on_advanced_setting_changed)
        self.chk_unsharp.stateChanged.connect(self.on_advanced_setting_changed)
        advanced_layout.addWidget(unsharp_row)

        # Sobel Edge Filter
        sobel_row, self.chk_sobel, self.slider_sobel, self.lbl_sobel_val = create_adv_row(
            "Sobel Edge Filter", 0, 100, 5, 50, "0.50"
        )
        self.slider_sobel.valueChanged.connect(lambda v: self.lbl_sobel_val.setText(f"{v/100.0:.2f}"))
        self.slider_sobel.sliderReleased.connect(self.on_advanced_setting_changed)
        self.chk_sobel.stateChanged.connect(self.on_advanced_setting_changed)
        advanced_layout.addWidget(sobel_row)

        # Canny Edge Filter
        canny_row, self.chk_canny, self.slider_canny, self.lbl_canny_val = create_adv_row(
            "Canny Edge Filter", 0, 100, 5, 50, "0.50"
        )
        self.slider_canny.valueChanged.connect(lambda v: self.lbl_canny_val.setText(f"{v/100.0:.2f}"))
        self.slider_canny.sliderReleased.connect(self.on_advanced_setting_changed)
        self.chk_canny.stateChanged.connect(self.on_advanced_setting_changed)
        advanced_layout.addWidget(canny_row)

        # User Toggle for Inverting Classes Dynamically
        self.chk_invert_yolo = QCheckBox("Invert YOLO/SAHI Classes (0=Negative, 1=Positive)")
        self.chk_invert_yolo.setStyleSheet("color: #334155; font-weight: 500;")
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
        
        # Wrap button layout in a widget to add margin
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

        self.lbl_class0 = QLabel("<b>Positive cells:</b> <span style='color: #8b0000;'>0</span>")
        self.lbl_class1 = QLabel("<b>Negative cells:</b> <span style='color: #00008b;'>0</span>")
        
        for lbl in (self.lbl_class0, self.lbl_class1):
            lbl.setFont(QFont("Consolas", 11))
            lbl.setStyleSheet("color: #334155;")
            layout_results.addWidget(lbl)
            
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
            "- Left click twice to add a <span style='color: blue;'>negative</span> annotation"
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
            "This app was built using Google Gemini and uses YOLO and SAHI.<br><br>"
            "<b>References:</b><ul>"
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
        self.btn_load.clicked.connect(self.on_load_clicked)
        self.btn_take_screenshot.clicked.connect(self.on_take_screenshot_clicked)
        self.btn_reset.clicked.connect(self.on_reset_clicked)
        self.btn_advanced.toggled.connect(self.advanced_container.setVisible)
        self.video_widget.edits_changed.connect(self.on_recalculate_clicked)

    def load_settings(self):
        settings_path = os.path.join("essential_files", "settings.json")
        if os.path.exists(settings_path):
            try:
                with open(settings_path, 'r') as f:
                    data = json.load(f)
                
                # Block signals to prevent processing multiple times during load
                self.chk_invert_yolo.blockSignals(True)
                self.chk_clahe.blockSignals(True)
                self.chk_unsharp.blockSignals(True)
                self.chk_gamma.blockSignals(True)
                self.chk_blur.blockSignals(True)
                self.chk_bilateral.blockSignals(True)
                self.chk_norm.blockSignals(True)
                self.chk_vahadane.blockSignals(True)
                self.chk_sobel.blockSignals(True)
                self.chk_canny.blockSignals(True)
                
                # Apply states
                self.chk_invert_yolo.setChecked(data.get("invert_classes", False))
                
                self.chk_clahe.setChecked(data.get("clahe_enabled", False))
                self.slider_clahe.setValue(data.get("clahe_value", 20))
                
                self.chk_unsharp.setChecked(data.get("unsharp_enabled", False))
                self.slider_unsharp.setValue(data.get("unsharp_value", 10))
                
                self.chk_gamma.setChecked(data.get("gamma_enabled", False))
                self.slider_gamma.setValue(data.get("gamma_value", 10))
                
                self.chk_blur.setChecked(data.get("blur_enabled", False))
                self.slider_blur.setValue(data.get("blur_value", 3))
                
                self.chk_bilateral.setChecked(data.get("bilateral_enabled", False))
                self.slider_bilateral.setValue(data.get("bilateral_value", 9))
                
                self.chk_norm.setChecked(data.get("norm_enabled", False))
                self.slider_norm.setValue(data.get("norm_value", 100))
                
                self.chk_vahadane.setChecked(data.get("vahadane_enabled", False))
                self.slider_vahadane.setValue(data.get("vahadane_value", 100))
                
                self.chk_sobel.setChecked(data.get("sobel_enabled", False))
                self.slider_sobel.setValue(data.get("sobel_value", 50))
                
                self.chk_canny.setChecked(data.get("canny_enabled", False))
                self.slider_canny.setValue(data.get("canny_value", 50))
                
                # Enable sliders if checked
                self.slider_gamma.setEnabled(self.chk_gamma.isChecked())
                self.slider_blur.setEnabled(self.chk_blur.isChecked())
                self.slider_bilateral.setEnabled(self.chk_bilateral.isChecked())
                self.slider_norm.setEnabled(self.chk_norm.isChecked())
                self.slider_vahadane.setEnabled(self.chk_vahadane.isChecked())
                self.slider_clahe.setEnabled(self.chk_clahe.isChecked())
                self.slider_unsharp.setEnabled(self.chk_unsharp.isChecked())
                self.slider_sobel.setEnabled(self.chk_sobel.isChecked())
                self.slider_canny.setEnabled(self.chk_canny.isChecked())
                
                # Update text labels
                self.lbl_gamma_val.setText(f"{self.slider_gamma.value()/10.0:.1f}")
                v = self.slider_blur.value()
                self.lbl_blur_val.setText(f"{v if v%2!=0 else v+1}")
                self.lbl_bilateral_val.setText(f"{self.slider_bilateral.value()}")
                self.lbl_norm_val.setText(f"{self.slider_norm.value()/100.0:.2f}")
                self.lbl_vahadane_val.setText(f"{self.slider_vahadane.value()/100.0:.2f}")
                self.lbl_clahe_val.setText(f"{self.slider_clahe.value()/10.0:.1f}")
                self.lbl_unsharp_val.setText(f"{self.slider_unsharp.value()/10.0:.1f}")
                self.lbl_sobel_val.setText(f"{self.slider_sobel.value()/100.0:.2f}")
                self.lbl_canny_val.setText(f"{self.slider_canny.value()/100.0:.2f}")

                # Unblock signals
                self.chk_invert_yolo.blockSignals(False)
                self.chk_clahe.blockSignals(False)
                self.chk_unsharp.blockSignals(False)
                self.chk_gamma.blockSignals(False)
                self.chk_blur.blockSignals(False)
                self.chk_bilateral.blockSignals(False)
                self.chk_norm.blockSignals(False)
                self.chk_vahadane.blockSignals(False)
                self.chk_sobel.blockSignals(False)
                self.chk_canny.blockSignals(False)
                
            except Exception as e:
                self.set_status(f"Error loading settings: {e}")

    def save_settings(self):
        os.makedirs("essential_files", exist_ok=True)
        settings_path = os.path.join("essential_files", "settings.json")
        data = {
            "invert_classes": self.chk_invert_yolo.isChecked(),
            "clahe_enabled": self.chk_clahe.isChecked(),
            "clahe_value": self.slider_clahe.value(),
            "unsharp_enabled": self.chk_unsharp.isChecked(),
            "unsharp_value": self.slider_unsharp.value(),
            "gamma_enabled": self.chk_gamma.isChecked(),
            "gamma_value": self.slider_gamma.value(),
            "blur_enabled": self.chk_blur.isChecked(),
            "blur_value": self.slider_blur.value(),
            "bilateral_enabled": self.chk_bilateral.isChecked(),
            "bilateral_value": self.slider_bilateral.value(),
            "norm_enabled": self.chk_norm.isChecked(),
            "norm_value": self.slider_norm.value(),
            "vahadane_enabled": self.chk_vahadane.isChecked(),
            "vahadane_value": self.slider_vahadane.value(),
            "sobel_enabled": self.chk_sobel.isChecked(),
            "sobel_value": self.slider_sobel.value(),
            "canny_enabled": self.chk_canny.isChecked(),
            "canny_value": self.slider_canny.value()
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
        self.chk_clahe.blockSignals(True)
        self.chk_unsharp.blockSignals(True)
        self.chk_gamma.blockSignals(True)
        self.chk_blur.blockSignals(True)
        self.chk_bilateral.blockSignals(True)
        self.chk_norm.blockSignals(True)
        self.chk_vahadane.blockSignals(True)
        self.chk_sobel.blockSignals(True)
        self.chk_canny.blockSignals(True)
        
        # Reset Checkboxes
        self.chk_invert_yolo.setChecked(False)
        self.chk_clahe.setChecked(False)
        self.chk_unsharp.setChecked(False)
        self.chk_gamma.setChecked(False)
        self.chk_blur.setChecked(False)
        self.chk_bilateral.setChecked(False)
        self.chk_norm.setChecked(False)
        self.chk_vahadane.setChecked(False)
        self.chk_sobel.setChecked(False)
        self.chk_canny.setChecked(False)
        
        # Reset Sliders
        self.slider_clahe.setValue(20)
        self.slider_unsharp.setValue(10)
        self.slider_gamma.setValue(10)
        self.slider_blur.setValue(3)
        self.slider_bilateral.setValue(9)
        self.slider_norm.setValue(100)
        self.slider_vahadane.setValue(100)
        self.slider_sobel.setValue(50)
        self.slider_canny.setValue(50)
        
        # Disable Sliders
        self.slider_gamma.setEnabled(False)
        self.slider_blur.setEnabled(False)
        self.slider_bilateral.setEnabled(False)
        self.slider_norm.setEnabled(False)
        self.slider_vahadane.setEnabled(False)
        self.slider_clahe.setEnabled(False)
        self.slider_unsharp.setEnabled(False)
        self.slider_sobel.setEnabled(False)
        self.slider_canny.setEnabled(False)
        
        # Update text labels
        self.lbl_gamma_val.setText("1.0")
        self.lbl_blur_val.setText("3")
        self.lbl_bilateral_val.setText("9")
        self.lbl_norm_val.setText("1.00")
        self.lbl_vahadane_val.setText("1.00")
        self.lbl_clahe_val.setText("2.0")
        self.lbl_unsharp_val.setText("1.0")
        self.lbl_sobel_val.setText("0.50")
        self.lbl_canny_val.setText("0.50")
        
        # Unblock signals
        self.chk_invert_yolo.blockSignals(False)
        self.chk_clahe.blockSignals(False)
        self.chk_unsharp.blockSignals(False)
        self.chk_gamma.blockSignals(False)
        self.chk_blur.blockSignals(False)
        self.chk_bilateral.blockSignals(False)
        self.chk_norm.blockSignals(False)
        self.chk_vahadane.blockSignals(False)
        self.chk_sobel.blockSignals(False)
        self.chk_canny.blockSignals(False)
        
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

    def on_recalculate_clicked(self):
        """Forces KPI and count recalculation based on manually added/deleted boxes in the viewer"""
        if self.last_screenshot_rgb is None or not hasattr(self.video_widget, 'boxes'):
            self.set_status("No active image to recalculate.")
            return

        current_opt = self.combo.currentText()
        boxes = self.video_widget.boxes
        
        c0 = sum(1 for b in boxes if b[4] == 0)
        c1 = sum(1 for b in boxes if b[4] == 1)
        counts = {0: c0, 1: c1}
        
        denom = c0 + c1
        pct = (c0 / denom * 100.0) if denom > 0 else 0.0
        
        if current_opt == "ER/PR analysis":
            kpi_text = f"ER/PR Positive Index: {pct:.1f}%"
        else:
            kpi_text = f"KI67 index : {pct:.1f}%"
            
        payload = {
            'class_counts': {f"class_{k}": int(v) for k, v in counts.items()},
            'kpi_text': kpi_text,
            'class_names': CLASS_NAMES.get(current_opt, ["Positive cells", "Negative cells"])
        }
        self.update_counts(payload)
        self.set_status("Cell counts recalculated based on manual UI edits.")

    def on_reset_clicked(self):
        self.last_screenshot_rgb = None
        self.video_widget.set_placeholder_text("Select an AI model to begin")
        self.update_frame(None)
        names = CLASS_NAMES.get(self.combo.currentText(), ["Positive cells", "Negative cells"])
        self.lbl_class0.setText(f"<b>{names[0]}:</b> <span style='color: #8b0000;'>0</span>")
        self.lbl_class1.setText(f"<b>{names[1]}:</b> <span style='color: #00008b;'>0</span>")
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
        h, w, _ = frame_rgb.shape
        
        current_opt = self.combo.currentText()
        invert_classes = self.chk_invert_yolo.isChecked()
        
        # Capture settings
        use_clahe = self.chk_clahe.isChecked()
        clahe_val = self.slider_clahe.value() / 10.0
        
        use_unsharp = self.chk_unsharp.isChecked()
        unsharp_val = self.slider_unsharp.value() / 10.0
        
        use_gamma = self.chk_gamma.isChecked()
        gamma_val = self.slider_gamma.value() / 10.0
        
        use_blur = self.chk_blur.isChecked()
        blur_val = self.slider_blur.value()
        if blur_val % 2 == 0: blur_val += 1
        
        use_bilateral = self.chk_bilateral.isChecked()
        bilateral_val = self.slider_bilateral.value()
        
        use_norm = self.chk_norm.isChecked()
        norm_val = self.slider_norm.value() / 100.0
        
        use_vahadane = self.chk_vahadane.isChecked()
        vahadane_val = self.slider_vahadane.value() / 100.0
        
        use_sobel = self.chk_sobel.isChecked()
        sobel_val = self.slider_sobel.value() / 100.0
        
        use_canny = self.chk_canny.isChecked()
        canny_val = self.slider_canny.value() / 100.0
        
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

        # Ensure active model label is up-to-date with actual fallback used
        self.update_active_model_label()

        def update_progress(pct, msg):
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(pct)
            self.set_status(f"{pct}% done. {msg}")
            QApplication.processEvents()

        self.btn_take_screenshot.setEnabled(False)
        self.btn_reset.setEnabled(False)

        # Run Universal CPU Inference with Enhancement Options
        drawn_rgb_or_bgr, counts, kpi_text, class_names, boxes_out = run_screenshot_inference(
            frame_rgb, model_to_use, current_opt, CLASS_NAMES, 
            progress_callback=update_progress, invert_classes=invert_classes,
            use_clahe=use_clahe, clahe_val=clahe_val,
            use_unsharp=use_unsharp, unsharp_val=unsharp_val,
            use_gamma=use_gamma, gamma_val=gamma_val,
            use_blur=use_blur, blur_val=blur_val,
            use_bilateral=use_bilateral, bilateral_val=bilateral_val,
            use_norm=use_norm, norm_val=norm_val,
            use_vahadane=use_vahadane, vahadane_val=vahadane_val,
            use_sobel=use_sobel, sobel_val=sobel_val,
            use_canny=use_canny, canny_val=canny_val
        )
        
        self.progress_bar.setVisible(False)
        self.progress_bar.setValue(0)
        
        bpl = 3 * w
        
        # Original Image (always RGB raw capture)
        qt_orig = QImage(frame_rgb.data, w, h, bpl, QImage.Format.Format_RGB888).copy()

        # The base Enhanced Image (without hardcoded boxes)
        qt_res = QImage(drawn_rgb_or_bgr.data, w, h, bpl, QImage.Format.Format_RGB888).copy()
        
        # Set images AND vector box coordinates directly to the UI Widget
        self.video_widget.set_images(qt_res, original_qimg=qt_orig, boxes=boxes_out)
        
        payload = {
            'class_counts': {f"class_{k}": int(v) for k, v in counts.items()},
            'kpi_text': kpi_text,
            'class_names': class_names
        }
        self.update_counts(payload)
        self.set_status("Capture analysis complete. Hover to pan/zoom, click to edit boxes.")
        self.btn_take_screenshot.setEnabled(True)
        self.btn_reset.setEnabled(True)

    def update_frame(self, qimage: QImage, orig_qimage: QImage = None):
        self.video_widget.set_images(qimage, orig_qimage)

    def update_counts(self, payload: dict):
        cn = payload.get('class_counts', {})
        names = payload.get('class_names', CLASS_NAMES.get(self.combo.currentText(), ["class_0","class_1"]))
        
        c0 = cn.get("class_0", 0)
        c1 = cn.get("class_1", 0)
        self.lbl_class0.setText(f"<b>{names[0]}:</b> <span style='color: #8b0000;'>{c0}</span>")
        self.lbl_class1.setText(f"<b>{names[1]}:</b> <span style='color: #00008b;'>{c1}</span>")

        self.lbl_kpi.setText(payload.get('kpi_text', ''))

    def set_status(self, text: str):
        self.lbl_status.setText(f"Status: {text}")

    def on_option_changed(self, option_text: str):
        try:
            names = CLASS_NAMES.get(option_text, ["class_0","class_1"])
            self.lbl_class0.setText(f"<b>{names[0]}:</b> <span style='color: #8b0000;'>0</span>")
            self.lbl_class1.setText(f"<b>{names[1]}:</b> <span style='color: #00008b;'>0</span>")
            self.lbl_kpi.setText("KPI: -")
            
            self._load_screenshot_models()
            self.update_active_model_label()
            
            if self.last_screenshot_rgb is not None:
                self.process_screenshot()
                
        except Exception as e:
            self.set_status(str(e))

    def on_load_clicked(self):
        self._load_screenshot_models()


def main():
    if hasattr(Qt.HighDpiScaleFactorRoundingPolicy, 'PassThrough'):
        QGuiApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
        
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()