import cv2
import dlib
import torch
import numpy as np
from torchvision import transforms

# Ensure these are defined as in your original code
from FF_train import MultiTaskResNet, get_label_mappings
from PIL import Image

# --- Load Model ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load label mappings
gender_mapping, race_mapping, age_mapping = get_label_mappings(
    "fairface_label_train.csv"
)

# Reverse mappings for display
gender_inv = {v: k for k, v in gender_mapping.items()}
race_inv = {v: k for k, v in race_mapping.items()}
age_inv = {v: k for k, v in age_mapping.items()}

# Transform
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# Load model
model = MultiTaskResNet(len(race_mapping), len(age_mapping)).to(device)
model.load_state_dict(torch.load("fairface_cnn_model.pth"))
model.eval()

# --- Setup Face Detector ---
detector = dlib.get_frontal_face_detector()


def get_face_with_padding(image, rect, pad=0.25):
    h, w = image.shape[:2]
    x1, y1, x2, y2 = rect.left(), rect.top(), rect.right(), rect.bottom()

    dw = int((x2 - x1) * pad)
    dh = int((y2 - y1) * pad)

    x1 = max(0, x1 - dw)
    y1 = max(0, y1 - dh)
    x2 = min(w, x2 + dw)
    y2 = min(h, y2 + dh)

    return image[y1:y2, x1:x2], (x1, y1, x2, y2)


def predict(image):
    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        gender_out, race_out, age_out = model(tensor)
        gender_prob = torch.sigmoid(gender_out.squeeze()).item()
        gender = int(gender_prob > 0.5)
        race = torch.argmax(race_out, dim=1).item()
        age = torch.argmax(age_out, dim=1).item()

    return gender_inv[gender], race_inv[race], age_inv[age]


# --- Start Webcam ---
cap = cv2.VideoCapture(0)
print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = detector(rgb_frame)

    for rect in faces:
        face_img, (x1, y1, x2, y2) = get_face_with_padding(frame, rect, pad=0.25)
        if face_img.size == 0:
            continue

        try:
            gender, race, age = predict(face_img)
            label = f"{gender}, {race}, {age}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
        except Exception as e:
            print("Prediction error:", e)

    cv2.imshow("Multi-task Prediction", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
