import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import random

class ChestXrayDataset(Dataset):
    def __init__(self, is_training_set, transform=None):
        self.is_training_set = is_training_set
        self.transform = transform

        self.image_paths = []
        self.labels = []

        """
        Join validation and test datasets as a single dataset because one of them has only 16 images.
        After joining, this is equally split into test and validation datasets.
        """
        if self.is_training_set == True:
            self.__get_images__('./data/train')
        else:
            self.__get_images__('./data/val')
            self.__get_images__('./data/test')

        if is_training_set == True:
            print(f"Training set, Total X-rays : {len(self.image_paths)}, Num. Healthy : {self.labels.count(0)}, Num. diseased : {self.labels.count(1)}")
        else:
            print(f"Test + Validation set, Total X-rays : {len(self.image_paths)}, Num. Healthy : {self.labels.count(0)}, Num. diseased : {self.labels.count(1)}")


    def __get_images__(self, image_dir):
        for label, category in enumerate(['NORMAL', 'PNEUMONIA']):
            print(f"Label : {label}, Category : {category}")
            category_path = os.path.join(image_dir, category)

            for filename in os.listdir(category_path):
                if filename.endswith(".jpeg") or filename.endswith(".png"):
                    self.image_paths.append(os.path.join(category_path, filename))
                    self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)


    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load the image using PIL
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)
        else:
            # If no transform, convert to tensor manually (i.e., ToTensor)
            image = transforms.ToTensor()(image)

        return image, label

    def show_images(self):
        normal_label_indexes = [i for i, value in enumerate(self.labels) if value == 0]
        pneumonia_label_indexes = [i for i, value in enumerate(self.labels) if value == 1]

        num_images = 5
        random_normal_indexes = random.sample(normal_label_indexes, num_images)
        random_pneumonia_indexes = random.sample(pneumonia_label_indexes, num_images)
        image_list = []

        for i in range(num_images):
            image_list.append(Image.open(self.image_paths[random_normal_indexes[i]]).convert("L"))
        for i in range(num_images):
            image_list.append(Image.open(self.image_paths[random_pneumonia_indexes[i]]).convert("L"))

        rows, cols = 2, num_images
        fig, axes = plt.subplots(rows, cols, figsize=(15, 8))
        # Loop through each axis and plot an image
        for i, ax in enumerate(axes.flat):
            if i < len(image_list):  # If there are fewer images than the grid size, stop plotting
                ax.imshow(image_list[i], cmap='gray')  # Display the image
                ax.axis('off')  # Turn off axis for cleaner presentation
 
                if i < cols:
                    ax.set_title("NORMAL", fontsize=12, color="blue")
                else:
                    ax.set_title("PNEUMONIA", fontsize=12, color="red")

            else:
                ax.axis('off')  # Hide the empty subplots if there are fewer images

        plt.tight_layout()  # Adjust layout to prevent overlap
        plt.show() 

#This is for debugging
if __name__ == "__main__":
    transform = transforms.Compose([
    transforms.Resize((224, 224)),            # Resize to fit model input size (e.g., 224x224 for ResNet)
    transforms.ToTensor(),                    # Convert image to a PyTorch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalization
    transforms.RandomHorizontalFlip()         # Random horizontal flip for data augmentation
    ])

    basic_transform = transforms.Compose([
    transforms.Resize((224, 224)),            # Resize to fit model input size (e.g., 224x224 for ResNet)
    transforms.ToTensor(),                    # Convert image to a PyTorch tensor
    ])

    # Create the dataset instance for train data
    train_dataset = ChestXrayDataset(is_training_set=False, transform=basic_transform)  # Without transforms for raw images
    train_dataset.show_images()
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    print(f"Dataset size : {len(train_loader.dataset)}")
