import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 데이터 변환 정의
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 데이터셋 로드
train_dataset = datasets.ImageFolder(root='C:/data/IMG/train', transform=transform)
val_dataset = datasets.ImageFolder(root='C:/data/IMG/val', transform=transform)
test_dataset = datasets.ImageFolder(root='C:/data/IMG/test', transform=transform)

# DataLoader 설정
batch_size = 64
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


# ResNet-18 모델 로드 및 마지막 레이어 교체
model = models.resnet18(pretrained=True)
for param in model.parameters():
    param.requires_grad = False

num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(train_dataset.classes))
model = model.to(device)

# VGG-16 모델 로드 및 마지막 레이어 교체
model = models.vgg16(pretrained=True)
for param in model.parameters():
    param.requires_grad = False

num_ftrs = model.classifier[6].in_features
model.classifier[6] = nn.Linear(num_ftrs, len(train_dataset.classes))
model = model.to(device)

# 손실 함수 및 옵티마이저 설정
learning_rate = 0.001
class_weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 0.5, 2.0, 2.0, 2.0, 2.0], device=device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)

# Custom CNN model
class ProposedCNN(nn.Module):
    def __init__(self):
        super(ProposedCNN, self).__init__()
        # Define sequential model: Convolution + ReLU + Pooling layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3),  # First convolutional layer
            nn.ReLU(),  # Activation function
            nn.Conv2d(16, 16, kernel_size=3, padding=1),  # Second convolutional layer
            nn.ReLU(),
            nn.MaxPool2d(2),  # First pooling layer
            # Additional convolutional and pooling layers
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),  # Apply dropout
            nn.AdaptiveAvgPool2d((6, 6))  # Apply adaptive pooling
        )
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 6 * 6, 512),  # First fully connected layer
            nn.ReLU(),
            nn.Linear(512, 9)  # Final output layer (number of classes = 9)
        )

    def forward(self, x):
        x = self.conv_layers(x)  # Pass through convolution layers
        x = x.view(-1, 128 * 6 * 6)  # Flatten the tensor
        x = self.fc_layers(x)  # Pass through fully connected layers
        return x


def validate(model, val_loader):
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            val_loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    val_loss /= len(val_loader)
    accuracy = 100 * correct / total
    return val_loss, accuracy

# Set model, loss function, and optimization algorithm
model = ProposedCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training process
num_epochs = 10  # Set number of epochs
# 에폭 별 평균 손실을 저장할 리스트 초기화
epoch_losses = []

# Hyperparameters
learning_rate = 0.0001  # Adjusted learning rate
batch_size = 64         # Adjusted batch size
num_epochs = 10         # Adjusted number of epochs

# Model instantiation with a custom learning rate
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Adjust data loaders with the new batch size
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# 리스트 초기화
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    # Calculate average losses and accuracy for training
    train_loss = running_loss / len(train_loader)
    train_accuracy = 100 * correct / total

    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)

    # Validate the model
    val_loss, val_accuracy = validate(model, val_loader)
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)

    print(f'Epoch {epoch + 1}: Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, '
          f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')

print('Finished Training')


# Evaluation process
model.eval()  # Set model to evaluation mode
correct = 0
total = 0
true_labels = []
predicted_labels = []
with torch.no_grad():  # Disable gradient calculation
    for i, (inputs, labels) in enumerate(test_loader, 1):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        true_labels.extend(labels.cpu().numpy())
        predicted_labels.extend(predicted.cpu().numpy())
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        if i % 10 == 0:  # Print accuracy every 10 batches
            print(f'Batch {i}, Current Accuracy: {100 * correct / total:.2f}%')

# Print final evaluation result
accuracy = 100 * correct / total
print(f'Test Accuracy: {accuracy:.2f}%')

# 클래스 이름 로드
class_names = train_dataset.classes

# 혼동행렬 계산
cm = confusion_matrix(true_labels, predicted_labels)

# 혼돈행렬을 확률로 변환하기 위해 각 행으로 나눔
cm_prob = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# 혼돈행렬 시각화 (확률로)
plt.figure(figsize=(10, 8))
sns.heatmap(cm_prob, annot=True, fmt=".2f", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix as Probabilities')
plt.show()

epochs = range(1, num_epochs + 1)

# Losses 그래프
plt.figure(figsize=(10, 5))
plt.plot(epochs, train_losses, 'r-', label='Training Loss')
plt.plot(epochs, val_losses, 'b-', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# Accuracies 그래프
plt.figure(figsize=(10, 5))
plt.plot(epochs, train_accuracies, 'r-', label='Training Accuracy')
plt.plot(epochs, val_accuracies, 'b-', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)
plt.show()





# 새로 제공된 혼동 행렬을 기반으로 계산
# confusion_matrix_prob_updated = np.array([
#     [0.99, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
#     [0.00, 0.99, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
#     [0.00, 0.00, 0.92, 0.01, 0.02, 0.00, 0.00, 0.03, 0.00],
#     [0.00, 0.00, 0.01, 0.96, 0.01, 0.00, 0.00, 0.01, 0.00],
#     [0.01, 0.01, 0.02, 0.01, 0.89, 0.00, 0.01, 0.08, 0.01],
#     [0.00, 0.00, 0.01, 0.00, 0.00, 0.98, 0.00, 0.00, 0.00],
#     [0.01, 0.02, 0.03, 0.00, 0.02, 0.01, 0.93, 0.00, 0.00],
#     [0.01, 0.01, 0.02, 0.01, 0.03, 0.00, 0.00, 0.93, 0.01],
#     [0.02, 0.00, 0.03, 0.00, 0.01, 0.00, 0.00, 0.12, 0.80]
# ])

confusion_matrix_prob_updated = np.array([
    [0.99, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    [0.00, 0.99, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    [0.01, 0.00, 0.94, 0.01, 0.01, 0.00, 0.00, 0.03, 0.00],
    [0.00, 0.00, 0.02, 0.97, 0.00, 0.00, 0.00, 0.01, 0.00],
    [0.02, 0.02, 0.04, 0.00, 0.82, 0.00, 0.01, 0.08, 0.01],
    [0.00, 0.00, 0.01, 0.00, 0.00, 0.98, 0.00, 0.00, 0.00],
    [0.00, 0.01, 0.03, 0.00, 0.01, 0.01, 0.93, 0.00, 0.00],
    [0.01, 0.01, 0.02, 0.01, 0.02, 0.00, 0.00, 0.93, 0.01],
    [0.01, 0.01, 0.04, 0.00, 0.01, 0.00, 0.00, 0.12, 0.80]
])


class_labels = ['Center', 'Dounut', 'Edge-Loc','Edge-Ring', 'LOC', 'Near-full', 'Random', 'Scratch','none']
# 각 메트릭을 계산합니다.
# TP는 대각선 값이며, FP는 해당 열의 다른 값들의 합, FN은 해당 행의 다른 값들의 합입니다.
# 각 클래스별로 TP, FP, FN, Precision, Recall, F1 Score를 계산합니다.
metrics_updated = {
    'TP': {label: 0 for label in class_labels},
    'FP': {label: 0 for label in class_labels},
    'FN': {label: 0 for label in class_labels},
    'TN': {label: 0 for label in class_labels},
    'Precision': {label: 0 for label in class_labels},
    'Recall': {label: 0 for label in class_labels},
    'F1': {label: 0 for label in class_labels}
}

# 각 클래스별 TP, FP, FN, TN 계산
for i in class_labels:
    TP = confusion_matrix_prob_updated[i, i] * num_samples_per_class
    FP = np.sum(confusion_matrix_prob_updated[:, i]) * num_samples_per_class - TP
    FN = np.sum(confusion_matrix_prob_updated[i, :]) * num_samples_per_class - TP
    TN = np.sum(confusion_matrix_prob_updated) * num_samples_per_class - (TP + FP + FN)
    
    metrics_updated['TP'][i] = TP
    metrics_updated['FP'][i] = FP
    metrics_updated['FN'][i] = FN
    metrics_updated['TN'][i] = TN

    # 계산된 TP, FP, FN을 이용하여 Precision, Recall 계산
    metrics_updated['Precision'][i] = TP / (TP + FP) if (TP + FP) > 0 else 0
    metrics_updated['Recall'][i] = TP / (TP + FN) if (TP + FN) > 0 else 0
    metrics_updated['F1'][i] = 2 * (metrics_updated['Precision'][i] * metrics_updated['Recall'][i]) / (metrics_updated['Precision'][i] + metrics_updated['Recall'][i]) if (metrics_updated['Precision'][i] + metrics_updated['Recall'][i]) > 0 else 0

# 전체 평균 메트릭 계산
average_precision_updated = np.mean(list(metrics_updated['Precision'].values()))
average_recall_updated = np.mean(list(metrics_updated['Recall'].values()))
average_f1_updated = np.mean(list(metrics_updated['F1'].values()))

average_precision_updated, average_recall_updated, average_f1_updated