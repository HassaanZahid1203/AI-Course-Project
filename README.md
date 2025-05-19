# FairFace Attribute Classifier

A multi-task deep learning application that classifies facial attributes (gender, race, and age) from images or webcam feed using PyTorch and PyQt5.

![Application Screenshot](screenshot.png) *(See "Adding Screenshots" section below for how to add your own screenshot)*

## Features

- Real-time face detection and attribute classification via webcam
- Image upload and analysis functionality
- Multi-task learning (gender, race, age) with confidence scores
- Clean PyQt5 interface with tabbed navigation
- Model evaluation metrics visualization

## Requirements

### Core Dependencies
- Python 3.8+
- PyTorch 1.12+
- TorchVision
- OpenCV
- PyQt5
- dlib
- pandas
- numpy
- Pillow
- matplotlib
- seaborn

### Optional (for development)
- Jupyter Notebook
- black (code formatting)
- flake8 (linting)

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/HassaanZahid1203/FairFace-Attribute-Classifier.git
   cd FairFace-Attribute-Classifier


Create and activate a virtual environment (recommended):

bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
Install dependencies:

bash
pip install -r requirements.txt
If you don't have a requirements.txt, install manually:

bash
pip install torch torchvision opencv-python pyqt5 dlib pandas numpy pillow matplotlib seaborn
For GPU support with PyTorch (CUDA):

bash
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu113
Download model weights and datasets:

Place fairface_cnn_model.pth in the root directory

Download the UTKFace dataset and extract to a data folder

Place fairface_label_train.csv and fairface_label_val.csv in the root directory

Usage
Running the Application
GUI Application:

bash
python MAIN.py
Or alternatively:

bash
python app.py
Webcam-only Version:

bash
python webcam.py
Application Modes
Webcam Tab:

Real-time face detection

Click "Capture & Analyze" to classify attributes

Results displayed below the webcam feed

Upload Tab:

Click "Upload Image" to select an image file

Click "Analyze" to classify attributes

Results displayed below the image

Training the Model
To retrain the model:

bash
python FF_train.py
Evaluating the Model
To evaluate model performance:

bash
python FF_test.py
To generate evaluation visualizations:

bash
python visualize_metrics.py
File Structure
FairFace-Attribute-Classifier/
├── app.py                # Main application (PyQt5)
├── MAIN.py               # Alternative main application
├── FF_train.py           # Model training script
├── FF_test.py            # Model evaluation script
├── webcam.py             # Webcam-only implementation
├── visualize_metrics.py  # Metrics visualization
├── fairface_cnn_model.pth  # Pretrained model weights
├── fairface_label_train.csv  # Training labels
├── fairface_label_val.csv    # Validation labels
├── training_dataset_evaluation.png    # Training metrics
├── validation_dataset_evaluation.png  # Validation metrics
└── README.md             # This file
Adding Screenshots
To add screenshots of your application:

Take a screenshot while the application is running

Save it as screenshot.png in the root directory

The image will automatically display in the README

Alternatively, you can:

Create a screenshots/ directory

Add your screenshots there

Reference them in the README like this:

markdown
![Webcam Tab](screenshots/webcam_tab.png)
![Upload Tab](screenshots/upload_tab.png)
Model Performance
The model achieves the following metrics on the validation set:

Attribute	Accuracy	Precision	Recall	F1-Score
Gender	92%	0.92	0.92	0.92
Race	65%	0.67	0.65	0.66
Age	76%	0.77	0.76	0.76
Detailed metrics are available in the generated PNG files:

training_dataset_evaluation.png

validation_dataset_evaluation.png

Troubleshooting
Webcam not working:

Ensure no other application is using the webcam

Check OpenCV installation with python -c "import cv2; print(cv2.__version__)"

Model loading errors:

Verify fairface_cnn_model.pth exists in the root directory

Check PyTorch installation with python -c "import torch; print(torch.__version__)"

Dlib installation issues:

On Windows, you may need to install CMake first: pip install cmake

On macOS: brew install cmake

Contributing
Contributions are welcome! Please fork the repository and create a pull request with your changes.

License
This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgments
UTKFace dataset providers

FairFace dataset for inspiration

PyTorch and PyQt5 communities
