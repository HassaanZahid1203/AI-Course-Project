# FairFace Attribute Classifier

## Table of Contents
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [File Structure](#file-structure)
- [Model Performance](#model-performance)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Features
- Real-time face detection and attribute classification
- Multi-task learning (gender, race, age)
- PyQt5 GUI interface
- Model evaluation metrics

## Requirements
### Core Dependencies
```bash
Python 3.8+
PyTorch
OpenCV
PyQt5
dlib
pandas
numpy
Pillow
matplotlib
seaborn
```

## Installation
1. Clone the repository:
```bash
git clone https://github.com/HassaanZahid1203/FairFace-Attribute-Classifier.git
cd FairFace-Attribute-Classifier
```

2. Set up virtual environment:
```bash
python -m venv venv
# Linux/Mac:
source venv/bin/activate
# Windows:
venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

If requirements.txt doesn't exist:
```bash
pip install torch torchvision opencv-python pyqt5 dlib pandas numpy pillow matplotlib seaborn
```

4. Download datasets:
* UTKFace Dataset
* Place CSV files in root directory:
```bash
data/
├── utkface/
│   ├── 1_0_0_20161219204523064.jpg
│   └── ...
├── fairface_label_train.csv
└── fairface_label_val.csv
```

## Usage
### Application Modes
1. Main GUI:
```bash
python MAIN.py
```

2. Webcam-only:
```bash
python webcam.py
# Press 'q' to quit
```

3. Training:
```bash
python FF_train.py
```

4. Evaluation:
```bash
python FF_test.py
python visualize_metrics.py
```

### Key Controls
| Function | Command |
|----------|---------|
| Webcam Capture | Click "Capture & Analyze" |
| Image Upload | Click "Upload Image" then "Analyze" |
| Exit | Close window or press 'q' |

## File Structure
```bash
FairFace-Attribute-Classifier/
├── app.py
├── MAIN.py
├── FF_train.py
├── FF_test.py
├── webcam.py
├── visualize_metrics.py
├── fairface_cnn_model.pth
├── fairface_label_train.csv
├── fairface_label_val.csv
├── requirements.txt
├── data/
└── README.md
```

## Model Performance
### Validation Metrics
| Task | Accuracy | Precision | Recall | F1-Score |
|------|----------|-----------|--------|----------|
| Gender | 92% | 0.92 | 0.92 | 0.92 |
| Race | 65% | 0.67 | 0.65 | 0.66 |
| Age | 76% | 0.77 | 0.76 | 0.76 |

## Troubleshooting
| Issue | Solution |
|-------|----------|
| Webcam not detected | Check permissions; try `cv2.VideoCapture(0)` |
| Model load errors | Verify `fairface_cnn_model.pth` exists |
| Dlib fails | Windows: `pip install cmake` first |
| CUDA OOM | Reduce batch size in FF_train.py |
| CSV errors | Check paths in FF_train.py |

## Contributing
1. Fork the repository
2. Create branch:
```bash
git checkout -b feature/NewFeature
```

3. Commit changes:
```bash
git commit -m 'Add new feature'
```

4. Push:
```bash
git push origin feature/NewFeature
```

5. Open PR

## License
MIT License

## Acknowledgments
* UTKFace dataset providers
* FairFace research team
* PyTorch/PyQt5 communities
* Kaggle
