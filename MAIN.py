import sys
import os
import cv2
import torch
import numpy as np
from PIL import Image
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QLabel, QPushButton, QTabWidget, 
                            QFileDialog, QMessageBox)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QTimer
from torchvision import transforms

# Import model classes from your training files
from FF_train import MultiTaskResNet, get_label_mappings

class FaceAttributeApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FairFace Attribute Classifier")
        self.setMinimumSize(800, 600)

        self.model = None
        self.device = None
        self.transform = None
        self.label_maps = None
        self.face_cascade = None
        self.cap = None
        self.current_frame = None
        self.original_image = None

        self.setup_ui()

        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        self.initialize_model()
        self.init_webcam()

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
        self.webcam_view = QLabel("Webcam Feed")
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
        self.tabs.addTab(upload_tab, "Upload Image")
        main_layout.addWidget(self.tabs)

        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

    def initialize_model(self):
        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            gender_mapping, race_mapping, age_mapping = get_label_mappings("fairface_label_train.csv")
            self.label_maps = (
                {v: k for k, v in gender_mapping.items()},
                {v: k for k, v in race_mapping.items()},
                {v: k for k, v in age_mapping.items()}
            )

            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                     std=[0.229, 0.224, 0.225])
            ])

            model = MultiTaskResNet(len(race_mapping), len(age_mapping)).to(self.device)
            model.load_state_dict(torch.load("fairface_cnn_model.pth", map_location=self.device))
            model.eval()
            self.model = model
            self.status_label.setText("Model loaded and ready")

        except Exception as e:
            self.status_label.setText(f"Error loading model: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to initialize model: {str(e)}")

    def init_webcam(self):
        self.cap = cv2.VideoCapture(0)
        if self.cap.isOpened():
            self.timer = QTimer()
            self.timer.timeout.connect(self.update_webcam)
            self.timer.start(30)
            self.status_label.setText("Webcam started - Model ready")
        else:
            self.status_label.setText("Warning: Webcam not available")
            self.webcam_view.setText("Webcam not available")

    def update_webcam(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        self.current_frame = frame
        display = frame.copy()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)

        self.display_image(display, self.webcam_view)

    def capture_and_analyze(self):
        if self.current_frame is None:
            return

        results = self.analyze_frame(self.current_frame)
        self.webcam_results.setText(results)

    def upload_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", "Images (*.png *.jpg *.jpeg *.bmp)"
        )
        if not path:
            return
            
        try:
            # Load the image
            img = cv2.imread(path)
            if img is None:
                QMessageBox.warning(self, "Error", "Failed to load image")
                return
            
            # Store original and working copies
            self.original_image = img.copy()
            self.current_frame = img.copy()
            
            # Process the image like webcam frames
            display = img.copy()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Use same face detection parameters as webcam
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.3,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            # Draw rectangles around detected faces (like webcam)
            for (x, y, w, h) in faces:
                cv2.rectangle(display, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Display the image with face rectangles
            self.display_image(display, self.upload_view)
            self.analyze_btn.setEnabled(True)
            self.status_label.setText("Image loaded - Faces detected: {}".format(len(faces)))
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to process image: {str(e)}")

    def enhance_image(self, image):
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        enhanced = cv2.filter2D(enhanced, -1, kernel)
        return enhanced

    def analyze_image(self):
        if self.current_frame is None:
            return
        results = self.analyze_frame(self.current_frame)
        self.upload_results.setText(results)

    def analyze_frame(self, frame):
        if self.model is None:
            return "Model not loaded"

        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Use consistent face detection parameters for both modes
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(100, 100),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        if len(faces) == 0:
            # Fallback to less strict parameters if no faces found
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.05,
                minNeighbors=3,
                minSize=(50, 50)
            )
            if len(faces) == 0:
                return "No faces detected"

        results = []
        reverse_gender, reverse_race, reverse_age = self.label_maps
        
        for (x, y, w, h) in faces:
            # Apply consistent padding (25% of face width/height)
            pad_w = int(w * 0.25)
            pad_h = int(h * 0.25)
            x1 = max(0, x - pad_w)
            y1 = max(0, y - pad_h)
            x2 = min(frame.shape[1], x + w + pad_w)
            y2 = min(frame.shape[0], y + h + pad_h)
            
            face = frame[y1:y2, x1:x2]
            
            if face.size == 0:
                continue
                
            try:
                # Convert to PIL Image with identical processing
                img = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
                
                # Apply high-quality resizing
                img = img.resize((224, 224), Image.BICUBIC)
                
                # Apply model transformations
                img_t = self.transform(img).unsqueeze(0).to(self.device)
                
                # Get predictions with confidence scores
                with torch.no_grad():
                    gender_out, race_out, age_out = self.model(img_t)
                    
                    # Process gender prediction
                    gender_prob = torch.sigmoid(gender_out).item()
                    gender = int(gender_prob > 0.5)
                    gender_conf = max(gender_prob, 1 - gender_prob)
                    
                    # Process race prediction
                    race_probs = torch.softmax(race_out, dim=1)
                    race_conf, race = torch.max(race_probs, dim=1)
                    
                    # Process age prediction
                    age_probs = torch.softmax(age_out, dim=1)
                    age_conf, age = torch.max(age_probs, dim=1)
                    
                    # Format results with confidence
                    result = (
                        f"Gender: {reverse_gender[gender]} ({gender_conf*100:.1f}%)\n"
                        f"Race: {reverse_race[race.item()]} ({race_conf.item()*100:.1f}%)\n"
                        f"Age: {reverse_age[age.item()]} ({age_conf.item()*100:.1f}%)"
                    )
                    
                    # Debug output
                    print(f"Gender raw score: {gender_out.item():.4f}, "
                        f"Probability: {gender_prob:.4f}, "
                        f"Prediction: {reverse_gender[gender]}")
                    
                    results.append(result)
                    
            except Exception as e:
                print(f"Prediction error: {str(e)}")
                continue
                
        return "\n\n".join(results) if results else "No predictions made"

    def display_image(self, image, widget):
        try:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            bytes_per_line = ch * w
            qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg).scaled(widget.width(), widget.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            widget.setPixmap(pixmap)
        except Exception as e:
            print(f"Error displaying image: {str(e)}")

    def closeEvent(self, event):
        if self.cap and self.cap.isOpened():
            self.cap.release()
        event.accept()

if __name__ == "__main__":
    try:
        app = QApplication(sys.argv)
        app.setStyle("Fusion")
        window = FaceAttributeApp()
        window.show()
        sys.exit(app.exec_())
    except Exception as e:
        QMessageBox.critical(None, "Error", f"Application error: {str(e)}")
        sys.exit(1)
