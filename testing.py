import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import time
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)

# Data loading and preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),  # Input: 3x32x32, Output: 32x32x32
            nn.ReLU(),
            nn.MaxPool2d(2, 2),             # Output: 32x16x16
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)              # Output: 64x8x8
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 64 * 8 * 8)
        x = self.fc(x)
        return x


model = SimpleCNN()  # Recreate the same model structure
model.load_state_dict(torch.load("cifar_model.pth"))
model.to(device)
model.eval()  # Set to evaluation mode
print("Model loaded and ready for inference!")



from PIL import Image

# Define CIFAR-10 class names
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

# Load and transform your own image
image = Image.open("chevrolet-corvette-zr1-coupe-001.jpg").resize((32, 32))  # Resize to CIFAR-10 input size
image = transform(image).unsqueeze(0).to(device)  # Apply same transform, add batch dimension

# Set model to evaluation mode
model.eval()
with torch.no_grad():
    output = model(image)
    _, predicted = torch.max(output, 1)

print(f"Predicted class: {classes[predicted.item()]}")
