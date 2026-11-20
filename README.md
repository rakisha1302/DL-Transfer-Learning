# DL- Developing a Neural Network Classification Model using Transfer Learning

## AIM
To develop an image classification model using transfer learning with VGG19 architecture for the given dataset.

## DESIGN STEPS
DESIGN STEPS
STEP 1:
Import required libraries and define image transforms.

STEP 2:
Load training and testing datasets using ImageFolder.

STEP 3:
Visualize sample images from the dataset.

STEP 4:
Load pre-trained VGG19, modify the final layer for binary classification, and freeze feature extractor layers.

STEP 5:
Define loss function (BCEWithLogitsLoss) and optimizer (Adam). Train the model and plot the loss curve.

STEP 6:
Evaluate the model with test accuracy, confusion matrix, classification report, and visualize predictions.


## PROGRAM

### Name:Rakisha R

### Register Number:212225230223

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchsummary import summary
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from torch.utils.data import DataLoader
from torchvision import models, datasets
import seaborn as sns
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])
dataset_path = "C:\\Users\\admin\\OneDrive\\Desktop\\py torch DL\\chip_data\\dataset" 
train_dataset = datasets.ImageFolder(root=f"{dataset_path}/train", transform=transform)
test_dataset = datasets.ImageFolder(root=f"{dataset_path}/test", transform=transform)
def show_sample_images(dataset, num_images=5):
    fig, axes = plt.subplots(1, num_images, figsize=(5, 5))
    for i in range(num_images):
        image, label = dataset[i]
        image = image.permute(1, 2, 0)  # Convert tensor format (C, H, W) to (H, W, C)
        axes[i].imshow(image)
        axes[i].set_title(dataset.classes[label])
        axes[i].axis("off")
    plt.show()
# Show sample images from the training dataset
show_sample_images(train_dataset)
print(f"Total number of training samples: {len(train_dataset)}")
print(f"Total number of testing samples: {len(test_dataset)}")
first_image,label =train_dataset[0]
print(f"Shape of the first image: {first_image.shape}")
train_loader = DataLoader(train_dataset,batch_size=32,shuffle=True)
test_loader = DataLoader(test_dataset,batch_size=32,shuffle=False)
num_classes = len(train_dataset.classes)
print("Number of classes:", num_classes)

model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)


for param in model.parameters():
    param.requires_grad = False

model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.classifier[6].parameters(), lr=0.001)

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=model.to(device)
from torchsummary import summary
summary(model,input_size=(3,224,224))

def train_model(model, train_loader,test_loader,num_epochs=10):
    train_losses = []
    val_losses = []
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_losses.append(running_loss / len(train_loader))

        # Compute validation loss
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

         

        val_losses.append(val_loss / len(test_loader))
        model.train()

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Validation Loss: {val_losses[-1]:.4f}')

    
    # Plot training and validation loss
    print("Name:Rakisha R")
    print("Register Number:212225230223")
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss', marker='o')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss', marker='s')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()
# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
## Step 4: Test the Model and Compute Confusion Matrix & Classification Report
def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = correct / total
    print(f'Test Accuracy: {accuracy:.4f}')

    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=train_dataset.classes, yticklabels=train_dataset.classes)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    # Print classification report
    print("Name:Rakisha R")
    print("Register Number:212225230223")
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=train_dataset.classes))

train_model(model, train_loader,test_loader)
test_model(model, test_loader)
def predict_image(model, image_index, dataset):
    model.eval()

    image, label = dataset[image_index]

    with torch.no_grad():
        image_tensor = image.unsqueeze(0).to(device)
        output = model(image_tensor)

        # Get the class with the highest score
        predicted = torch.argmax(output, dim=1).item()

    class_names = dataset.classes

    image_to_display = transforms.ToPILImage()(image)

    plt.figure(figsize=(4, 4))
    plt.imshow(image_to_display)
    plt.title(
        f'Actual: {class_names[label]}\n'
        f'Predicted: {class_names[predicted]}'
    )
    plt.axis("off")
    plt.show()

    print(f'Actual: {class_names[label]}, Predicted: {class_names[predicted]}')
predict_image(model, image_index=65, dataset=test_dataset)

predict_image(model, image_index=25, dataset=test_dataset)
l




```

### OUTPUT



<img width="777" height="711" alt="image" src="https://github.com/user-attachments/assets/196f7d60-3774-4ee9-a889-732bab0597ad" />

<img width="776" height="832" alt="image" src="https://github.com/user-attachments/assets/3ad2110c-939f-4e79-acab-453fc5c2d249" />

<img width="782" height="702" alt="image" src="https://github.com/user-attachments/assets/a4fc1edc-5166-41e8-904f-f0aabdc8151b" />

<img width="547" height="215" alt="image" src="https://github.com/user-attachments/assets/4c60fe2c-a3e8-4900-aeaf-2549ae063e94" />

<img width="441" height="502" alt="image" src="https://github.com/user-attachments/assets/2e3ff275-1529-4f19-81a4-2d47aca73e51" />

<img width="432" height="505" alt="image" src="https://github.com/user-attachments/assets/7ff5f749-18c7-4d6d-b55a-d94efd5f237f" />

## RESULT
Thus the python program to develop an image classification model using transfer learning with VGG19 architecture is executed successfully.
