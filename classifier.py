import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision.models import ResNet18_Weights
import matplotlib.pyplot as plt
import time
from chest_xray_model import chest_xray_model

from ChestXrayDataset import ChestXrayDataset


HELP_DESCRIPTION = f"""Classifies chest X-ray images as healthy or having pneumonia"""

def get_pretrained_resnet18_model():
    """
    Use pre-trained ResNET model and modify it's final layer to classify x-ray images
    """
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT) 
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2) # 2 output classes for healthy and unhealthy cases

    # Freeze all the layers of the pre-trained model
    for param in model.parameters():
        param.requires_grad = False  

    # Unfreeze the final layer for training
    for param in model.fc.parameters():
        param.requires_grad = True

    return model


def get_extended_resnet18_model():
    """
    Use a more advanced model that has an additional convolution layer after pre-trained ResNET
    and a new fully connected layer for x-ray image classification
    """
    model = chest_xray_model()

    return model

def load_data():
    """
    Load the data. Generate training, validation and test datasets.
    Apply basic transformation to the data to feed it to ResNet18
    """
    transform = transforms.Compose([
    transforms.Resize((224, 224)),            # Resize to fit model input size (e.g., 224x224 for ResNet)
    transforms.ToTensor(),                    # Convert image to a PyTorch tensor
    transforms.Grayscale(num_output_channels=3),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize images based on how ResNet18 is trained
    transforms.RandomHorizontalFlip()         # Random horizontal flip for data augmentation
    ]) 

    train_dataset = ChestXrayDataset(is_training_set=True, transform=transform)
    print(f"Showing some images from training data")
    train_dataset.show_images()
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    unseen_dataset = ChestXrayDataset(is_training_set=False, transform=transform)
    val_dataset_size = int(0.5 * len(unseen_dataset))
    test_dataset_size = len(unseen_dataset) - val_dataset_size

    val_dataset, test_dataset = random_split(unseen_dataset, [val_dataset_size, test_dataset_size])
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    print(f"Size of training set : {len(train_loader.dataset)}")
    print(f"Size of validation set : {len(val_loader.dataset)}")
    print(f"Size of test set : {len(test_loader.dataset)}")
    
    return train_loader, val_loader, test_loader 


def plot_losses(model_name, num_epochs, train_losses, val_losses, final_test_accuracy):
    """
    Plot the training and validation losses.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss', color='blue', marker='o')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss', color='red', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Test set accuracy : {final_test_accuracy}')
    plt.legend()
    plot_file_name = f'chest_xray_loss_plot_{model_name}_{num_epochs}_epochs.png'
    plt.savefig(plot_file_name)
    plt.close()

    print(f"Training and validation losses plot saved to : {plot_file_name}")


def get_model_evaluation(model, criterion, data, device):
    """
    Compute model evaluation against the validation dataset.
    Return average loss and average accuracy on the dataset
    """
    model.eval()
    correct = 0
    total = 0
    val_loss = 0
    with torch.no_grad():
        for inputs, labels in data:
           inputs, labels = inputs.to(device), labels.to(device)
           outputs = model(inputs)
           loss = criterion(outputs, labels)
           val_loss += loss.item()
           _, predicted = torch.max(outputs, 1)
           total += labels.size(0)
           correct += (predicted == labels).sum().item()

    val_accuracy = correct / total
    avg_val_loss = val_loss / len(data)
    return avg_val_loss, val_accuracy


def train_model(model, model_name, num_epochs, criterion, train_data_loader, val_data_loader, test_data_loader, device):
    """
    Train the model. Train on the training set, check its evaluation against the validation set.
    Choose the model with the lowest losses on the validation set.
    """    
    best_val_loss = float('inf')
    best_model = None
    training_losses = []
    training_accuracy = []
    validation_losses = []
    validation_accuracy = []
    
    for epoch in range(num_epochs):
        print(f"Training epoch : {epoch}")
        start_time = time.time()
        model.train()
        current_loss = 0.0
   
        for inputs, labels in train_data_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
           
            optimizer.step()
            current_loss += loss.item()

        train_loss = current_loss / len(train_data_loader) * 1.0
        val_loss, val_accuracy = get_model_evaluation(model, criterion, val_data_loader, device)
        print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')
        training_losses.append(train_loss)
        validation_losses.append(val_loss)
        validation_accuracy.append(val_accuracy)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict()
        
        end_time = time.time()
        epoch_training_time = (end_time - start_time) / 60.0
        print(f"Training epoch {epoch} took : {epoch_training_time:.2f} minutes")

    # Get the final accuracy of the model on the test set
    model.load_state_dict(best_model)
    final_test_loss, final_test_accuracy = get_model_evaluation(model, criterion, test_data_loader, device)
    print(f'\n\nFinal Accuracy on Test Set: {final_test_accuracy}')

    plot_losses(model_name, num_epochs, training_losses, validation_losses, final_test_accuracy)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=HELP_DESCRIPTION, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--numepochs', help="Number of epochs to train the model for", default=15)
    parser.add_argument('--modeltype', choices=['pretrained_resnet18', 'modified_resnet18'], help="Choose either pretrained Resnet18 or pre-trained Resnet18 with additional convolution layers")

    args = parser.parse_args()
    num_epochs = int(args.numepochs)
    model_type = args.modeltype

    print(f"Model type : {model_type}, Number of epochs : {num_epochs}")

    if model_type == 'pretrained_resnet18':
        model = get_pretrained_resnet18_model()
    else:
        model = get_extended_resnet18_model()
 
    print(model)

    optimizer = optim.Adam(model.fc.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    train_loader, val_loader, test_loader = load_data()

    torch.manual_seed(7)
    train_model(model, model_type, num_epochs, criterion, train_loader, val_loader, test_loader, device)
