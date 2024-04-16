from tqdm import trange
import torch
import os
import torch.nn as nn
import time
import copy
import numpy as np
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix, roc_auc_score, precision_recall_fscore_support
import math
import torch.nn.functional as F

# Figure properties.
import seaborn as sns
import matplotlib.pyplot as plt
column_width_pt = 516.0
pt_to_inch = 1 / 72.27
column_width_inches = column_width_pt * pt_to_inch
aspect_ratio = 4 / 3
sns.set(style="whitegrid", font_scale=2, rc={"figure.figsize": (column_width_inches, column_width_inches / aspect_ratio)})
sns.set_context("paper")


def select_trainer():
    pass

def independent_trainer():
    pass

def sequential_trainer():
    pass

def joint_trainer():
    pass

def train_model(model, data_loaders, learning_rate, weights, num_epochs, early_stopping, warmup_epoch, models_dir, device, freeze_encoder=False, to_disk=True):
    weight_concept_loss = weights['concept']
    weight_task_loss = weights['task']
    weight_rec_loss = weights['rec']
    weight_lat_loss = weights['lat']

    if freeze_encoder:
        for param in model.concept_encoder.parameters():
            param.requires_grad = False

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion_c = nn.MSELoss()
    criterion_y = nn.CrossEntropyLoss()
    criterion_rec = nn.MSELoss()
    criterion_lat = nn.MSELoss()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = np.Inf

    history = {
        'train_concept_loss': [],
        'train_task_loss': [],
        'train_rec_loss': [],
        'train_lat_loss': [],
        'train_total_loss': [],
        'val_concept_loss': [],
        'val_task_loss': [],
        'val_rec_loss': [],
        'val_lat_loss': [],
        'val_total_loss': [],
    }

    epochs_no_improve = 0
    best_epoch = 0
    early_stop = False

    since = time.time()
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_task_loss = 0.0
            running_concept_loss = 0.0
            running_rec_loss = 0.0
            running_lat_loss = 0.0
            running_loss = 0.0

            # Iterate over data.
            for sample in tqdm(data_loaders[phase]):

                imgs, concepts, labels = sample['image'].to(device), sample['concepts'].to(device), sample['label'].to(device)
                optimizer.zero_grad()

                # forward
                with torch.set_grad_enabled(phase == 'train'):

                    preds_concepts, preds_task, preds_img_tilde, preds_concepts_tilde = model(imgs)
                    _, preds = torch.max(preds_task, 1)

                    loss = torch.tensor(0.0).to(device)

                    concept_loss = criterion_c(preds_concepts, concepts)
                    loss += (weight_concept_loss * concept_loss)

                    task_loss = criterion_y(preds_task, labels)
                    loss += (weight_task_loss * task_loss)

                    rec_loss = criterion_rec(preds_img_tilde, imgs)
                    loss += (weight_rec_loss * rec_loss)

                    lat_loss = criterion_lat(preds_concepts_tilde, concepts)
                    loss += (weight_lat_loss * lat_loss)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_task_loss += task_loss.item()
                running_concept_loss += concept_loss.item()
                running_rec_loss += rec_loss.item()
                running_lat_loss += lat_loss.item()
                running_loss += loss.item()

            epoch_task_loss = running_task_loss / len(data_loaders[phase])
            epoch_concept_loss = running_concept_loss / len(data_loaders[phase])
            epoch_rec_loss = running_rec_loss / len(data_loaders[phase])
            epoch_lat_loss = running_lat_loss / len(data_loaders[phase])
            epoch_loss = running_loss / len(data_loaders[phase])

            # update history
            if phase == 'train':
                history['train_task_loss'].append(epoch_task_loss)
                history['train_concept_loss'].append(epoch_concept_loss)
                history['train_rec_loss'].append(epoch_rec_loss)
                history['train_lat_loss'].append(epoch_lat_loss)
                history['train_total_loss'].append(epoch_loss)
            else:
                history['val_task_loss'].append(epoch_task_loss)
                history['val_concept_loss'].append(epoch_concept_loss)
                history['val_rec_loss'].append(epoch_rec_loss)
                history['val_lat_loss'].append(epoch_lat_loss)
                history['val_total_loss'].append(epoch_loss)

            print(
                f"Phase: {phase} - Epoch: {epoch + 1}, "
                f"Loss: {epoch_loss:.4f}, "
                f"Task Loss: {epoch_task_loss:.4f}, "
                f"Concept Loss: {epoch_concept_loss:.4f}, "
                f"Rec Loss: {epoch_rec_loss:.4f}, "
                f"Lat Loss: {epoch_lat_loss:.4f}."
            )

            # deep copy the model
            if phase == 'val':

                if epoch > warmup_epoch:
                    if epoch_loss < best_loss:
                        best_epoch = epoch
                        best_loss = epoch_loss
                        best_model_wts = copy.deepcopy(model.state_dict())
                        epochs_no_improve = 0
                    else:
                        epochs_no_improve += 1
                        # Trigger early stopping
                        if epochs_no_improve >= early_stopping:
                            print(f'\nEarly Stopping! Total epochs: {epoch}%')
                            early_stop = True
                            break

            if early_stop:
                break

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best epoch: {:0f}'.format(best_epoch))
    print('Best val loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)

    # Save model
    if to_disk:
        torch.save(model.state_dict(), os.path.join(models_dir, f"model_best.pt"))

    # Format history
    history = pd.DataFrame.from_dict(history, orient='index').transpose()

    return model, history

def evaluate(model, data_loader, device):

    all_preds = []
    all_preds_concepts = []
    all_labels = []
    all_concepts = []
    all_rec = []
    all_lat = []

    model.eval()
    with torch.no_grad():

        for sample in tqdm(data_loader):
            inputs, concepts, labels = sample['image'].to(device), sample['concepts'].to(device), sample['label'].to(device)
            all_labels.extend(labels.cpu().numpy())
            all_concepts.extend(concepts.cpu().numpy())

            # Prediction
            preds_concepts, preds_task, preds_rec, preds_lat = model(inputs)
            _, preds = torch.max(preds_task, 1)

            all_preds.extend(preds.cpu().numpy())
            all_preds_concepts.extend(preds_concepts.cpu().numpy())
            all_rec.append(F.mse_loss(preds_rec, inputs).item())
            all_lat.append(F.mse_loss(preds_lat, concepts).item())

    all_labels = np.array(all_labels)
    all_concepts = np.array(all_concepts)
    all_preds = np.array(all_preds)
    all_preds_concepts = np.array(all_preds_concepts)

    acc, prec, rec, spec, f1, gmean = compute_metrics(all_labels, all_preds)
    concept_mse = np.mean(np.square(all_concepts - all_preds_concepts))
    recon_mse = np.mean(all_rec)
    latent_mse = np.mean(all_lat)

    history = {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'specificity': spec,
        'f1_score': f1,
        'g_mean': gmean,
        'concept_mse': concept_mse,
        'recon_mse': recon_mse,
        'latent_mse': latent_mse
    }
    # Return a dataframe.
    history = pd.DataFrame.from_dict(history, orient='index').transpose()

    return history


def predict(model, data_loader, device):

    all_preds_concepts = []
    model.eval()
    with torch.no_grad():

        for sample in tqdm(data_loader):
            inputs, _, _ = sample['image'].to(device), sample['concepts'].to(device), sample['label'].to(device)

            preds_concepts, _, _, _ = model(inputs)
            all_preds_concepts.extend(preds_concepts.cpu().numpy())

    all_preds_concepts = np.array(all_preds_concepts)
    return all_preds_concepts

def compute_metrics(y_test_real, y_pred):
    cm = confusion_matrix(y_test_real, y_pred)
    accuracy = np.trace(cm) / float(np.sum(cm))

    tp = cm[0, 0]
    fp = np.sum(cm[:, 0]) - tp
    fn = np.sum(cm[0, :]) - tp
    tn = np.sum(cm) - (fp + fn + tp)
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0

    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
    g_mean = np.sqrt(recall * specificity)

    return accuracy, precision, recall, specificity, f1, g_mean

def plot_training(history, plot_training_dir, loss_names, plot_name_loss='Loss', plot_name_acc='Acc'):

    # Training results Loss function
    plt.figure()
    for c in loss_names:
        plt.plot(history[c], label=c)
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Losses')

    # Delete white space
    plt.tight_layout()
    plt.savefig(os.path.join(plot_training_dir, f"{plot_name_loss}.pdf"),  dpi=400, format='pdf')
    plt.show()

    # Training results Accuracy
    if 'train_acc' in history.columns and 'val_acc' in history.columns:
        plt.figure()
        for c in ['train_acc', 'val_acc']:
            plt.plot(100 * history[c], label=c)
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')

        # Delete white space
        plt.tight_layout()
        plt.savefig(os.path.join(plot_training_dir, f"{plot_name_acc}.pdf"),  dpi=400, format='pdf')
        plt.show()

def plot_reconstructions(plot_training_dir, model, data_loader, device, num_images=5):

    from PIL import Image
    model.eval()
    with torch.no_grad():
        for sample in data_loader:
            x = sample['image'].to(device)
            # Assuming the model returns reconstruction in the third position
            _, _, x_tilde, _ = model(x)
            break  # Only take the first batch

    # To numpy.
    x = x.cpu().numpy()[:num_images]
    x_tilde = x_tilde.cpu().numpy()[:num_images]

    # Convert to 0-255 range.
    x = (x * 255).astype(np.uint8).transpose(0, 2, 3, 1)
    x_tilde = (x_tilde * 255).astype(np.uint8).transpose(0, 2, 3, 1)

    x_h = np.concatenate(x, axis=1)
    x_tilde_h = np.concatenate(x_tilde, axis=1)
    x_out = np.concatenate((x_h, x_tilde_h), axis=0)

    x_out = Image.fromarray(x_out[:, :, 0])
    x_out.save(os.path.join(plot_training_dir, "img_rec.pdf"), resolution=400)