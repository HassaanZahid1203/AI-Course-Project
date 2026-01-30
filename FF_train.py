import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image


# Custom Dataset Class for FairFace
class FairFaceDataset(Dataset):
    def __init__(
        self,
        image_folder,
        csv_file,
        gender_mapping,
        race_mapping,
        age_mapping,
        transform=None,
    ):
        self.image_folder = image_folder
        self.data = pd.read_csv(csv_file)
        self.gender_mapping = gender_mapping
        self.race_mapping = race_mapping
        self.age_mapping = age_mapping
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        filename = row["file"]
        image_path = os.path.join(self.image_folder, filename)
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        gender = self.gender_mapping[row["gender"]]
        race = self.race_mapping[row["race"]]
        age = self.age_mapping[row["age"]]
        return image, gender, race, age


# Function to Create Label Mappings from Training CSV
def get_label_mappings(train_csv):
    data = pd.read_csv(train_csv)
    gender_mapping = {"Male": 0, "Female": 1}
    race_mapping = {race: idx for idx, race in enumerate(
        sorted(data["race"].unique()))}
    age_mapping = {age: idx for idx, age in enumerate(
        sorted(data["age"].unique()))}
    return gender_mapping, race_mapping, age_mapping


# Multi-Task CNN Model Based on ResNet50
class MultiTaskResNet(nn.Module):
    def __init__(self, num_race_classes, num_age_classes):
        super(MultiTaskResNet, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        self.resnet.fc = nn.Identity()  # Remove original fully connected layer
        # Single output for binary classification
        self.gender_head = nn.Linear(2048, 1)
        self.race_head = nn.Linear(2048, num_race_classes)
        self.age_head = nn.Linear(2048, num_age_classes)

    def forward(self, x):
        features = self.resnet(x)
        gender_out = self.gender_head(features)
        race_out = self.race_head(features)
        age_out = self.age_head(features)
        return gender_out, race_out, age_out


# Training Function
def train(
    model,
    train_loader,
    criterion_gender,
    criterion_race,
    criterion_age,
    optimizer,
    device,
):
    model.train()
    running_loss = 0.0
    for images, genders, races, ages in train_loader:
        images = images.to(device)
        # Convert to float for BCEWithLogitsLoss
        genders = genders.to(device).float()
        races = races.to(device)
        ages = ages.to(device)

        optimizer.zero_grad()
        gender_out, race_out, age_out = model(images)
        gender_loss = criterion_gender(gender_out.squeeze(), genders)
        race_loss = criterion_race(race_out, races)
        age_loss = criterion_age(age_out, ages)
        total_loss = gender_loss + race_loss + age_loss
        total_loss.backward()
        optimizer.step()
        running_loss += total_loss.item()
    return running_loss / len(train_loader)


# Validation Function
def validate(
    model, val_loader, criterion_gender, criterion_race, criterion_age, device
):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for images, genders, races, ages in val_loader:
            images = images.to(device)
            genders = genders.to(device).float()
            races = races.to(device)
            ages = ages.to(device)
            gender_out, race_out, age_out = model(images)
            gender_loss = criterion_gender(gender_out.squeeze(), genders)
            race_loss = criterion_race(race_out, races)
            age_loss = criterion_age(age_out, ages)
            total_loss = gender_loss + race_loss + age_loss
            running_loss += total_loss.item()
    return running_loss / len(val_loader)


# Main Function
def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Hyperparameters
    batch_size = 32
    learning_rate = 0.001
    num_epochs = 10

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get label mappings from training CSV
    gender_mapping, race_mapping, age_mapping = get_label_mappings(
        "fairface_label_train.csv"
    )

    # Define image transformations
    train_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225]),
        ]
    )
    val_transform = transforms.Compose(
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
        transform=train_transform,
    )
    val_dataset = FairFaceDataset(
        ".",
        "fairface_label_val.csv",
        gender_mapping,
        race_mapping,
        age_mapping,
        transform=val_transform,
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    num_race_classes = len(race_mapping)
    num_age_classes = len(age_mapping)
    model = MultiTaskResNet(num_race_classes, num_age_classes).to(device)

    # Define loss functions
    criterion_gender = nn.BCEWithLogitsLoss()
    criterion_race = nn.CrossEntropyLoss()
    criterion_age = nn.CrossEntropyLoss()

    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        train_loss = train(
            model,
            train_loader,
            criterion_gender,
            criterion_race,
            criterion_age,
            optimizer,
            device,
        )
        val_loss = validate(
            model, val_loader, criterion_gender, criterion_race, criterion_age, device
        )
        
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    # Save the trained model
    torch.save(model.state_dict(), "fairface_cnn_model.pth")
    print("Model saved as 'fairface_cnn_model.pth'")


if __name__ == "__main__":
    main()
