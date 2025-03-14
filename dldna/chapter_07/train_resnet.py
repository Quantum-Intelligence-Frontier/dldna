import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from dldna.chapter_07.resnet import ResNet18

def train_resnet18(epochs=10, batch_size=128, learning_rate=0.001):
    """ResNet-18 model training function."""
    
    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Data preprocessing
    transform = transforms.Compose([
        transforms.Resize(224),  # Resize to ResNet input size
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Load FashionMNIST dataset
    train_dataset = datasets.FashionMNIST(root='../../data', train=True,
                                        download=True, transform=transform)
    test_dataset = datasets.FashionMNIST(root='../../data', train=False,
                                       transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize ResNet-18 model (modify in_channels=1 for grayscale images)
    model = ResNet18(in_channels=1, num_classes=10).to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], '
                f'Loss: {running_loss/100:.4f}, '
                f'Acc: {100.*correct/total:.2f}%')
        running_loss = 0.0
        
        # Evaluate with the test set at each epoch
        model.eval()
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()
        
        print(f'Epoch [{epoch+1}/{epochs}] Test Accuracy: {100.*test_correct/test_total:.2f}%')
    
    return model


def save_model(model, path='./tmp/models/resnet18_fashion.pth'):
    """Save the trained model."""
    torch.save(model.state_dict(), path)

def get_trained_model_and_test_image(model_path='./tmp/models/resnet18_fashion.pth'):
    """Function to load the trained model and infer with a test image"""
    # Load the model
    model = ResNet18(in_channels=1, num_classes=10)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')), strict=False) #Added map_location and strict=False
    model.eval()

    # Load FashionMNIST dataset
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to ResNet input size
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    test_dataset = datasets.FashionMNIST(
        root='../../data', 
        train=False,
        download=True,
        transform=transform
    )

    # Define class labels
    classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    # Get the first test image and label
    test_image, label = test_dataset[0]
    
    # Pass the image through the model
    with torch.no_grad():
        output = model(test_image.unsqueeze(0))
        pred = output.argmax(dim=1).item()
    
    # print(f"Actual class: {classes[label]} (Label: {label})")
    # print(f"Predicted class: {classes[pred]} (Label: {pred})")
    
    return model, test_image, label, pred, classes

if __name__ == "__main__":
    # Train the model
    model = train_resnet18(epochs=10)
    # Save the model
    save_model(model)