"""
Splits for Camelyon.
"""
import copy
import os
import sys

sys.path.extend([
    "./",
])
import seaborn as sns

from torch.utils.data import Dataset
from torchvision import transforms
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
from tqdm import tqdm

from src.utils import util_path

column_width_pt = 516.0
pt_to_inch = 1 / 72.27
column_width_inches = column_width_pt * pt_to_inch
aspect_ratio = 4 / 3
sns.set(style="whitegrid", font_scale=1.6, rc={"figure.figsize": (column_width_inches, column_width_inches / aspect_ratio)})

class dSpritesDataset(Dataset):
    def __init__(self, data_dir, fname, use_concepts=False):
        """
        Args:
            data_dir (string): Directory with all the images.
            fname (string): Name of the file with the data.
            use_concepts (bool): Whether to use the concepts as features.
        """

        self.data_dir = data_dir
        self.img_size = 64

        data = np.load(os.path.join(self.data_dir, fname), allow_pickle=True)
        self.images = data['X']
        self.labels = data['Y']
        self.use_concepts = use_concepts
        if self.use_concepts:
            try:
                self.concepts = data['G']
            except KeyError:
                self.use_concepts = False

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        im = np.expand_dims(image, 0) # Add channel dimension.
        im = np.repeat(im, 3, axis=0) # Repeat the channel 3 times.
        im = torch.from_numpy(im).float() # Convert to float.

        label = self.labels[idx]
        label = np.array(label).astype(np.int64)

        if self.use_concepts:
            c = self.concepts[idx]
            c = np.array(c).astype(np.float32)
            return {'image': im, 'label': label, 'concepts': c}
        else:
            return {'image': im, 'label': label}

if __name__ == "__main__":

    # Directories.
    dataset_name = 'dsprites'
    interim_dir = os.path.join('data', dataset_name, 'interim')
    reports_dir = os.path.join('reports', dataset_name)
    util_path.create_dir(reports_dir)

    # Parameters.
    num_epochs = 5
    learning_rate = 0.001
    n_classes = 2

    # Dataset.
    train_dataset = dSpritesDataset(data_dir=interim_dir, fname='dsprites_leakage_train.npz', use_concepts=True)
    val_dataset = dSpritesDataset(data_dir=interim_dir, fname='dsprites_leakage_val.npz', use_concepts=True)
    test_dataset = dSpritesDataset(data_dir=interim_dir, fname='dsprites_leakage_test.npz', use_concepts=True)

    # Creating DataLoaders.
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # # Iterator.
    dataiter = iter(train_loader)
    sample = next(dataiter)

    # Load the network.
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, n_classes)

    # Train.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Training loop using tqdm
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}")
        model.train()
        running_loss = 0.0

        for sample in tqdm(train_loader):
            inputs, labels = sample['image'].to(device), sample['label'].to(device)
            if train_dataset.use_concepts:
                concepts = sample['concepts'].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}")

    # Evaluation loop.
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():

        for sample in tqdm(test_loader):
            inputs, labels = sample['image'].to(device), sample['label'].to(device)
            if test_dataset.use_concepts:
                concepts = sample['concepts'].to(device)
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
    print(f"Accuracy: {accuracy}\nPrecision: {precision}\nRecall: {recall}\nF1-Score: {f1}")


    print("May the force be with you!")