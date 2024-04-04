import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Transformations applied to the dataset: resize, convert to tensor, normalize
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize images
])

# Loading training data
# Path needs to be set according to your environment
train_dataset = datasets.ImageFolder(root='C:/data/IMG/train',
                                     transform=transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

# Transformations for the test dataset
transform_test = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to match model input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Loading test dataset
# Path needs to be set according to your environment
test_dataset = datasets.ImageFolder(root='C:/data/IMG/test',
                                    transform=transform_test)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)


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


# Set model, loss function, and optimization algorithm
model = ProposedCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training process
num_epochs = 5  # Set number of epochs
for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader, 1):
        inputs, labels = inputs.to(device), labels.to(device)  # Move data to the appropriate device
        optimizer.zero_grad()  # Reset gradients
        outputs = model(inputs)  # Calculate model output
        loss = criterion(outputs, labels)  # Calculate loss
        loss.backward()  # Perform backpropagation
        optimizer.step()  # Update weights
        running_loss += loss.item()
        if i % 10 == 0:  # Print loss every 10 batches
            print(f'Epoch {epoch + 1}/{num_epochs}, Batch {i}, Loss: {running_loss / 10:.4f}')
            running_loss = 0.0

print('Finished Training')

# Print total number of batches for training and testing
print(f'Total training batches: {len(train_loader)}')
print(f'Total testing batches: {len(test_loader)}')

# Evaluation process
model.eval()  # Set model to evaluation mode
correct = 0
total = 0
with torch.no_grad():  # Disable gradient calculation
    for i, (inputs, labels) in enumerate(test_loader, 1):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        if i % 10 == 0:  # Print accuracy every 10 batches
            print(f'Batch {i}, Current Accuracy: {100 * correct / total:.2f}%')

# Print final evaluation result
accuracy = 100 * correct / total
print(f'Test Accuracy: {accuracy:.2f}%')
