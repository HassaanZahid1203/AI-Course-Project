import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np
from FF_train import FairFaceDataset, MultiTaskResNet, get_label_mappings


def test_model(model, data_loader, device):
    model.eval()
    gender_preds, gender_true = [], []
    race_preds, race_true = [], []
    age_preds, age_true = [], []

    with torch.no_grad():
        for images, genders, races, ages in data_loader:
            images = images.to(device)
            genders = genders.to(device).float()
            races = races.to(device)
            ages = ages.to(device)

            gender_out, race_out, age_out = model(images)

            # Gender predictions (binary)
            gender_pred = torch.sigmoid(gender_out.squeeze()).cpu().numpy()
            gender_pred = (gender_pred > 0.5).astype(int)
            gender_preds.extend(gender_pred)
            gender_true.extend(genders.cpu().numpy().astype(int))

            # Race predictions
            race_pred = torch.argmax(race_out, dim=1).cpu().numpy()
            race_preds.extend(race_pred)
            race_true.extend(races.cpu().numpy())

            # Age predictions
            age_pred = torch.argmax(age_out, dim=1).cpu().numpy()
            age_preds.extend(age_pred)
            age_true.extend(ages.cpu().numpy())

    return (gender_true, gender_preds), (race_true, race_preds), (age_true, age_preds)


def print_metrics(true_labels, pred_labels, label_mapping, task_name):
    if task_name == "Age":
        adjusted_preds = []
        for t, p in zip(true_labels, pred_labels):
            adjusted_preds.append(t if abs(t - p) <= 1 else p)
        eval_true = true_labels
        eval_pred = adjusted_preds
        print(f"\n{task_name} Metrics (with ±1 adjacency):")
    else:
        eval_true = true_labels
        eval_pred = pred_labels
        print(f"\n{task_name} Metrics:")

    # Accuracy
    accuracy = accuracy_score(eval_true, eval_pred)
    print(f"Accuracy: {accuracy:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(eval_true, eval_pred)
    print("\nConfusion Matrix:")
    print(cm)

    # Classification Report
    report = classification_report(
        eval_true,
        eval_pred,
        target_names=[str(k) for k in label_mapping.keys()],
        zero_division=0,
    )
    print("\nClassification Report:")
    print(report)


def main():
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get label mappings
    gender_mapping, race_mapping, age_mapping = get_label_mappings(
        "fairface_label_train.csv"
    )

    # Define transformations (same as validation in training)
    test_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225]),
        ]
    )

    # Create datasets
    train_dataset = FairFaceDataset(
        ".",
        "fairface_label_train.csv",
        gender_mapping,
        race_mapping,
        age_mapping,
        transform=test_transform,
    )
    val_dataset = FairFaceDataset(
        ".",
        "fairface_label_val.csv",
        gender_mapping,
        race_mapping,
        age_mapping,
        transform=test_transform,
    )

    # Create data loaders
    batch_size = 32
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    num_race_classes = len(race_mapping)
    num_age_classes = len(age_mapping)
    model = MultiTaskResNet(num_race_classes, num_age_classes).to(device)

    # Load trained weights
    model.load_state_dict(torch.load("fairface_cnn_model.pth"))
    print("Loaded model weights from 'fairface_cnn_model.pth'")

    # Test on training dataset
    print("\n=== Training Dataset Evaluation ===")
    (gender_true, gender_preds), (race_true, race_preds), (age_true, age_preds) = (
        test_model(model, train_loader, device)
    )

    print_metrics(gender_true, gender_preds, gender_mapping, "Gender")
    print_metrics(race_true, race_preds, race_mapping, "Race")
    print_metrics(age_true, age_preds, age_mapping, "Age")

    # Test on validation dataset
    print("\n=== Validation Dataset Evaluation ===")
    (gender_true, gender_preds), (race_true, race_preds), (age_true, age_preds) = (
        test_model(model, val_loader, device)
    )

    print_metrics(gender_true, gender_preds, gender_mapping, "Gender")
    print_metrics(race_true, race_preds, race_mapping, "Race")
    print_metrics(age_true, age_preds, age_mapping, "Age")


if __name__ == "__main__":
    main()
