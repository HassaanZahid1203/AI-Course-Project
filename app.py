import sys
import os
import traceback
import cv2
import numpy as np
from PIL import Image
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTabWidget,
    QFileDialog,
    QMessageBox,
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QTimer

# Try to import torch-related modules and FairFace model
try:
    import torch
    import torch.nn as nn
    from torchvision import transforms
    from FF_train import MultiTaskResNet, get_label_mappings

    TORCH_AVAILABLE = True
    print("PyTorch and FairFace model available")
except ImportError:
    TORCH_AVAILABLE = False
    print("Required modules not available - running in limited mode")


class SimpleAttributeApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Simple Facial Attribute Extractor")
        self.setMinimumSize(800, 600)

        # Initialize variables
        self.model = None
        self.device = None
        self.transform = None
        self.label_maps = None
        self.face_cascade = None
        self.cap = None
        self.current_frame = None

        # Setup UI
        try:
            self.setup_ui()
        except Exception as e:
            print(f"Error setting up UI: {e}")
            traceback.print_exc()
            sys.exit(1)

        # Initialize face detector
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        if self.face_cascade.empty():
            self.status_label.setText("Warning: Face detection not available")

        # Initialize webcam
        self.cap = cv2.VideoCapture(0)
        if self.cap.isOpened():
            self.timer = QTimer()
            self.timer.timeout.connect(self.update_webcam)
            self.timer.start(30)
        else:
            self.status_label.setText("Warning: Webcam not available")
            self.webcam_view.setText("Webcam not available")
            self.cap = None

        # Initialize model
        if TORCH_AVAILABLE:
            try:
                self.initialize_model()
            except Exception as e:
                print(f"Error initializing model: {e}")
                traceback.print_exc()
                self.status_label.setText("Error: Model initialization failed")
        else:
            self.status_label.setText("Limited mode: Model not available")

    def setup_ui(self):
        main_widget = QWidget()
        main_layout = QVBoxLayout()

        self.status_label = QLabel("Initializing...")
        self.status_label.setStyleSheet("color: blue; font-weight: bold;")
        main_layout.addWidget(self.status_label)

        self.tabs = QTabWidget()
        webcam_tab = QWidget()
        upload_tab = QWidget()

        # Webcam tab
        webcam_layout = QVBoxLayout()
        self.webcam_view = QLabel("Initializing webcam...")
        self.webcam_view.setAlignment(Qt.AlignCenter)
        self.webcam_view.setMinimumHeight(400)
        self.webcam_view.setStyleSheet("border: 1px solid #ccc;")
        webcam_layout.addWidget(self.webcam_view)

        webcam_controls = QHBoxLayout()
        self.capture_btn = QPushButton("Capture & Analyze")
        self.capture_btn.clicked.connect(self.capture_and_analyze)
        webcam_controls.addWidget(self.capture_btn)
        webcam_layout.addLayout(webcam_controls)

        self.webcam_results = QLabel("Predictions will appear here")
        self.webcam_results.setAlignment(Qt.AlignCenter)
        self.webcam_results.setStyleSheet(
            "background-color: #f0f0f0; padding: 10px; border-radius: 5px;"
        )
        self.webcam_results.setMinimumHeight(100)
        webcam_layout.addWidget(self.webcam_results)

        webcam_tab.setLayout(webcam_layout)

        # Upload tab
        upload_layout = QVBoxLayout()
        self.upload_view = QLabel("No image loaded")
        self.upload_view.setAlignment(Qt.AlignCenter)
        self.upload_view.setMinimumHeight(400)
        self.upload_view.setStyleSheet("border: 1px solid #ccc;")
        upload_layout.addWidget(self.upload_view)

        upload_controls = QHBoxLayout()
        self.upload_btn = QPushButton("Upload Image")
        self.upload_btn.clicked.connect(self.upload_image)
        self.analyze_btn = QPushButton("Analyze")
        self.analyze_btn.clicked.connect(self.analyze_image)
        self.analyze_btn.setEnabled(False)
        upload_controls.addWidget(self.upload_btn)
        upload_controls.addWidget(self.analyze_btn)
        upload_layout.addLayout(upload_controls)

        self.upload_results = QLabel("Predictions will appear here")
        self.upload_results.setAlignment(Qt.AlignCenter)
        self.upload_results.setStyleSheet(
            "background-color: #f0f0f0; padding: 10px; border-radius: 5px;"
        )
        self.upload_results.setMinimumHeight(100)
        upload_layout.addWidget(self.upload_results)

        upload_tab.setLayout(upload_layout)

        self.tabs.addTab(webcam_tab, "Webcam")
        self.tabs.addTab(upload_tab, "Upload")
        main_layout.addWidget(self.tabs)

        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

    def initialize_model(self):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        # Load label mappings
        label_csv = "fairface_label_train.csv"
        gender_map, race_map, age_map = get_label_mappings(label_csv)
        reverse_gender = {v: k for k, v in gender_map.items()}
        reverse_race = {v: k for k, v in race_map.items()}
        reverse_age = {v: k for k, v in age_map.items()}
        self.label_maps = (reverse_gender, reverse_race, reverse_age)

        # Transformation
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        # Load model
        num_race = len(race_map)
        num_age = len(age_map)
        self.model = MultiTaskResNet(num_race, num_age).to(self.device)
        state = torch.load("fairface_cnn_model.pth", map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()
        self.status_label.setText("Model loaded and ready")

    def update_webcam(self):
        if not self.cap:
            return
        ret, frame = self.cap.read()
        if not ret:
            return
        self.current_frame = frame
        display = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        for x, y, w, h in faces:
            cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
        rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        self.webcam_view.setPixmap(
            QPixmap.fromImage(qimg).scaled(
                self.webcam_view.width(), self.webcam_view.height(), Qt.KeepAspectRatio
            )
        )

    def capture_and_analyze(self):
        if self.current_frame is None:
            return
        # Save image
        cv2.imwrite("captured_image.jpg", self.current_frame)
        # Analyze frame
        results = self.run_inference(self.current_frame)
        self.webcam_results.setText(results)

    def upload_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", "Images (*.png *.jpg *.jpeg *.bmp)"
        )
        if not path:
            return
        img = cv2.imread(path)
        if img is None:
            self.upload_results.setText("Failed to load image")
            return
        self.current_frame = img
        display = img.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        for x, y, w, h in faces:
            cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
        rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        self.upload_view.setPixmap(
            QPixmap.fromImage(qimg).scaled(
                self.upload_view.width(), self.upload_view.height(), Qt.KeepAspectRatio
            )
        )
        self.analyze_btn.setEnabled(True)

    def analyze_image(self):
        if self.current_frame is None:
            return
        result = self.run_inference(self.current_frame)
        self.upload_results.setText(result)

    def run_inference(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        if len(faces) == 0:
            return "No faces detected"
        rg, rr, ra = self.label_maps
        texts = []
        for x, y, w, h in faces:
            face = frame[y: y + h, x: x + w]
            img = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
            inp = self.transform(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                g_out, r_out, a_out = self.model(inp)
                g = torch.sigmoid(g_out).item() > 0.5
                r = torch.argmax(r_out, dim=1).item()
                a = torch.argmax(a_out, dim=1).item()
            texts.append(f"Gender: {rg[int(g)]}, Race: {rr[r]}, Age: {ra[a]}")
        return "\n".join(texts)

    def closeEvent(self, event):
        if self.cap and self.cap.isOpened():
            self.cap.release()
        event.accept()


def show_error_dialog(message):
    msg = QMessageBox()
    msg.setIcon(QMessageBox.Critical)
    msg.setText("Error starting application")
    msg.setInformativeText(message)
    msg.setWindowTitle("Application Error")
    msg.exec_()


if __name__ == "__main__":
    try:
        sys.excepthook = lambda exctype, value, tb: show_error_dialog(
            str(value))
        app = QApplication(sys.argv)
        app.setStyle("Fusion")
        window = SimpleAttributeApp()
        window.show()
        sys.exit(app.exec_())
    except Exception as e:
        traceback.print_exc()
        show_error_dialog(str(e))
