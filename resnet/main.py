import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms
from my_resnet import resnet50_2way
from FakedditDataset import FakedditImageDataset, my_collate
from tqdm import tqdm
import os

# Data transformations
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Dataset paths and initializations
csv_dir = "../multimodal_only_samples/"
img_dir = "../multimodal_only_samples/images/"
csv_fname = 'multimodal_train.tsv'
train_dataset = FakedditImageDataset(os.path.join(
    csv_dir, csv_fname), img_dir, transform=data_transforms)

# DataLoader - Optimized for GPU
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=64,  # Increased batch size for GPU
                                               shuffle=True, num_workers=4,  # Increased num_workers for faster data loading
                                               collate_fn=my_collate)

# Device configuration - Using the first available GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Model initialization
model_ft = resnet50_2way(pretrained=True).to(device)

# Ensure the final layer's parameters require gradients
for param in model_ft.fc.parameters():
    param.requires_grad = True

# Loss and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer_ft = optim.Adam(model_ft.parameters(), lr=1e-4)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

# Training function


def train_model(model, criterion, optimizer, scheduler, num_epochs=1):
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        model.train()  # Set model to training mode

        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in tqdm(train_dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                # Ensure output dimensions match labels
                outputs = model(inputs).squeeze()
                # Get predictions from logits
                preds = torch.sigmoid(outputs) > 0.5
                loss = criterion(outputs, labels.float())

                loss.backward()
                optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels)

        scheduler.step()

        epoch_loss = running_loss / len(train_dataloader.dataset)
        epoch_acc = running_corrects.double() / len(train_dataloader.dataset)

        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    return model


# Train the model
model_ft = train_model(model_ft, criterion, optimizer_ft,
                       exp_lr_scheduler, num_epochs=20)

# Save the trained model
torch.save(model_ft.state_dict(), 'fakeddit_resnet_epochs20_full_train.pt')
torch.save(model_ft, "resnet_model_save_epochs20_full_train")
