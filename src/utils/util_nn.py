#%%
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

#%%
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
    weight_orth_loss = weights['orth']

    if freeze_encoder:
        for param in model.concept_encoder.parameters():
            param.requires_grad = False

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion_c = nn.MSELoss()      # Loss dei concetti
    criterion_y = nn.CrossEntropyLoss()# Loss classifcation
    criterion_rec = nn.MSELoss()     # Loss generation
    criterion_lat = nn.MSELoss()    
    
    # TODO: Loss ortogonalità di tutti i concetti sia supervised che unsupervised. 
    # Orthogonalità dei concetti supervised e unsupervised ma non tra tutti. 
    # Possible to build two mse_dictionary during training for different labels, as activation should change within label

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = np.Inf

    history = {
        'train_concept_loss': [],
        'train_task_loss': [],
        'train_rec_loss': [],
        'train_lat_loss': [],
        'train_orth_loss': [],
        'train_total_loss': [],
        'val_concept_loss': [],
        'val_task_loss': [],
        'val_rec_loss': [],
        'val_lat_loss': [],
        'val_orth_loss': [],
        'val_total_loss': []
    }

    epoch_mse_errors_dict = {}
    
    epochs_no_improve = 0
    best_epoch = 0
    early_stop = False

    since = time.time()
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        # Build distances dictionary of dictionaries
        epoch_mse_errors_dict[epoch] = {}

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
            running_orth_loss = 0.0
            running_loss = 0.0
            running_sup_unsup_dist = {}
            # Iterate over data.
            for sample in tqdm(data_loaders[phase]):

                imgs, concepts, labels = sample['image'].to(device), sample['concepts'].to(device), sample['label'].to(device)
                optimizer.zero_grad()

                # forward
                with torch.set_grad_enabled(phase == 'train'):
                    # print(imgs.size())
                    preds_concepts, unsup_concepts, preds_task, preds_img_tilde, preds_concepts_tilde, linear_weights = model(imgs)
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
                    
                    # orth_loss = cosine_similarity_loss(linear_weights)
                    orth_loss = orth_frob_loss(linear_weights)
                    # orth_loss = orth_gram_loss(linear_weights)
                    loss += (weight_orth_loss * orth_loss)
                    
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                # Calculate pairwise MSE (average over batch dimension)
                for i in range(preds_concepts.size(1)):
                        for j in range(unsup_concepts.size(1)):
                            if (i, j) not in running_sup_unsup_dist:
                                running_sup_unsup_dist[(i, j)] = 0.0
                            running_sup_unsup_dist[(i, j)] += F.mse_loss(preds_concepts[:, i], unsup_concepts[:, j]).item()

                # Statistics
                running_task_loss += task_loss.item()
                running_concept_loss += concept_loss.item()
                running_rec_loss += rec_loss.item()
                running_lat_loss += lat_loss.item()
                running_orth_loss += orth_loss.item()
                running_loss += loss.item()

            epoch_task_loss = running_task_loss / len(data_loaders[phase])
            epoch_concept_loss = running_concept_loss / len(data_loaders[phase])
            epoch_rec_loss = running_rec_loss / len(data_loaders[phase])
            epoch_lat_loss = running_lat_loss / len(data_loaders[phase])
            epoch_orth_loss = running_orth_loss / len(data_loaders[phase])
            epoch_loss = running_loss / len(data_loaders[phase])
            
            # Average over total number of batches
            for key in running_sup_unsup_dist:
                running_sup_unsup_dist[key] /= len(data_loaders[phase])
            epoch_mse_errors_dict[epoch][phase] = running_sup_unsup_dist
            
            # update history
            if phase == 'train':
                history['train_task_loss'].append(epoch_task_loss)
                history['train_concept_loss'].append(epoch_concept_loss)
                history['train_rec_loss'].append(epoch_rec_loss)
                history['train_lat_loss'].append(epoch_lat_loss)
                history['train_orth_loss'].append(epoch_orth_loss)
                history['train_total_loss'].append(epoch_loss)
            else:
                history['val_task_loss'].append(epoch_task_loss)
                history['val_concept_loss'].append(epoch_concept_loss)
                history['val_rec_loss'].append(epoch_rec_loss)
                history['val_lat_loss'].append(epoch_lat_loss)
                history['val_orth_loss'].append(epoch_orth_loss)
                history['val_total_loss'].append(epoch_loss)

            print(
                f"Phase: {phase} - Epoch: {epoch + 1}, "
                f"Loss: {epoch_loss:.4f}, "
                f"Task Loss: {epoch_task_loss:.4f}, "
                f"Concept Loss: {epoch_concept_loss:.4f}, "
                f"Rec Loss: {epoch_rec_loss:.4f}, "
                f"Lat Loss: {epoch_lat_loss:.4f}, "
                f"Orth Loss: {epoch_lat_loss:.4f}."
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

    return model, history, epoch_mse_errors_dict
#%%

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
            preds_concepts, unsup_concepts, preds_task, preds_rec, preds_lat, _ = model(inputs)
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

            preds_concepts, _, _, _, _, _ = model(inputs)
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
            _, _, _, x_tilde, _, _ = model(x)
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
    
######################### RICCARDO MSE PAIRWISE ########################
def mse_error_pairwise_batch(tensor1, tensor2):
    """
    Calculate the MSE between each value in tensor1 and all values in tensor2 for batches.
    
    Returns:
        torch.Tensor: Averaged MSE errors over the batch dimension, same size of Tensor1
    """
    batch_size, n1 = tensor1.shape
    _, n2 = tensor2.shape

    errors = torch.zeros(batch_size, n1, n2).to(tensor1.device)

    for i in range(n1):
        for j in range(n2):
            errors[:, i, j] = F.mse_loss(tensor1[:, i], tensor2[:, j], reduction='none')
            # print(f"MSE between tensor1[:, {i}] and tensor2[:, {j}]:")
            # print(errors[:, i, j])
            # print("-------------")
    
    return errors.mean(dim=0)
    
def plot_sup_unsup_mse_epoch(mse_errors_dict, preds_concepts_size, unsup_concepts_size, save_dir):
    num_epochs = len(mse_errors_dict)
    for i in range(preds_concepts_size):
        plt.figure()
        for j in range(unsup_concepts_size):
            train_values = [mse_errors_dict[epoch]['train'].get((i, j), None) for epoch in range(num_epochs)]
            val_values = [mse_errors_dict[epoch]['val'].get((i, j), None) for epoch in range(num_epochs)]
            plt.plot(range(1, num_epochs + 1), train_values, label=f'Concept {j + 1}', linestyle='-', color=f'C{j}')
            plt.plot(range(1, num_epochs + 1), val_values, linestyle='--', color=f'C{j}')

        plt.xlabel('Epochs')
        plt.ylabel('MSE')
        #plt.ylim(0, 0.05)   FIX A TRESHOLD for visualization purposes
        plt.title(f'Preds_concept {i + 1}')
        plt.legend(title='Legend:', loc='best', fontsize='small')
        plt.text(0.5, 0.92, 'Continuous line: Train\n-- line: Validation', horizontalalignment='center', 
                 verticalalignment='center', fontsize='small',
                 transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.5))
        plt.savefig(os.path.join(save_dir, f'mse_epoch_preds_concept_{i + 1}.png'))
        # plt.show()
        plt.close()

#################################################### correlations a distances post taining ############################
import pickle
from itertools import combinations
from scipy.stats import pearsonr, spearmanr
def calculate_and_save_distances(model, data_loaders, device, output_dir):
    model.eval()  # Set model to evaluation mode
    
    distance_dict = {}

    for phase in ['train', 'val', 'test']:
        count = 0
        for idx, sample in tqdm(enumerate(data_loaders[phase]), desc=f'Calculating distances for {phase}'):
            if count >= 400:
                break
            
            imgs, concepts = sample['image'].to(device), sample['concepts'].to(device)

            with torch.no_grad():
                preds_concepts, unsup_concepts, _, _, _, _ = model(imgs)

                sample_distances = np.zeros((preds_concepts.size(1), unsup_concepts.size(1)))

                for i in range(preds_concepts.size(1)):
                    for j in range(unsup_concepts.size(1)):
                        # Batch size = 1 so reduction is none
                        mse_value = F.mse_loss(preds_concepts[:, i], unsup_concepts[:, j], reduction='none').item()
                        sample_distances[i, j] = mse_value

            distance_dict[(phase, idx)] = sample_distances
            
            count += 1
        # print(f'DIST DICT: -> {np.shape(distance_dict[(phase, idx-1)])}')
    
    # Save distances to file
    distance_file_path = os.path.join(output_dir, 'distances.pkl')
    with open(distance_file_path, 'wb') as f:
        pickle.dump(distance_dict, f)


    return print(f'Distances saved to {distance_file_path}')

def plot_mse_values(distance_file_path, save_dir=None):
    
    if save_dir is None:
        save_dir = distance_file_path
    # Load distances from the file
    with open(distance_file_path, 'rb') as f:
        distance_dict = pickle.load(f)

    # Extract the first (phase, idx) tuple from the dictionary to get the number of concepts
    first_key = list(distance_dict.keys())[0]
    n_pred_concepts, n_unsup_concepts = distance_dict[first_key].shape

    # Prepare the figure
    fig, axs = plt.subplots(n_pred_concepts, figsize=(10, 6 * n_pred_concepts))
    if n_pred_concepts == 1:
        axs = [axs]

    # Plot for each prediction concept
    for i in range(n_pred_concepts):
        mse_values = np.array([distances[i] for distances in distance_dict.values()])
        
        # Calculate mean MSE values across all samples for the current prediction concept
        mean_mse_values = mse_values.mean(axis=0)
        
        x_labels = [f'Unsup Concept {j+1}' for j in range(n_unsup_concepts)]
        
        bar_width = 0.35
        x = np.arange(len(x_labels))
        
        axs[i].bar(x, mean_mse_values, bar_width, label=f'Preds Concept {i}', color='blue')
        
        axs[i].set_title(f'MSE Values for Preds Concept {i}')
        axs[i].set_ylabel('Mean MSE')
        axs[i].set_xticks(x)
        axs[i].set_xticklabels(x_labels, rotation=45, ha='right')
        axs[i].grid(axis='y', linestyle='--', alpha=0.7)
        axs[i].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'mse_preds_concept_{i + 1}.png'))
    plt.show()
    plt.close()
    
    return print(f'Plot saved to {save_dir}')
    
    
def calculate_and_save_correlations(model, data_loaders, device, output_dir):
    model.eval()  # Set model to evaluation mode

    preds_concepts_list = []
    unsup_concepts_list = []

    for phase in ['train', 'val', 'test']:
        count = 0
        for idx, sample in tqdm(enumerate(data_loaders[phase]), desc=f'Extracting concepts for {phase}'):
            if count >= 1000:
                break
            
            imgs, concepts = sample['image'].to(device), sample['concepts'].to(device)

            with torch.no_grad():
                preds_concepts, unsup_concepts, _, _, _, _ = model(imgs)

                preds_concepts_list.append(preds_concepts.cpu().numpy())
                unsup_concepts_list.append(unsup_concepts.cpu().numpy())
            
            count += 1

    preds_concepts_array = np.concatenate(preds_concepts_list, axis=0)
    unsup_concepts_array = np.concatenate(unsup_concepts_list, axis=0)

    n_pred_concepts = preds_concepts_array.shape[1]
    n_unsup_concepts = unsup_concepts_array.shape[1]

    correlations = {}

    for i in range(n_pred_concepts):
        pearson_values = []
        spearman_values = []
        for j in range(n_unsup_concepts):
            # Pearson correlation
            pearson_corr, _ = pearsonr(preds_concepts_array[:, i], unsup_concepts_array[:, j])
            # Spearman correlation
            spearman_corr, _ = spearmanr(preds_concepts_array[:, i], unsup_concepts_array[:, j])
            pearson_values.append(pearson_corr)
            spearman_values.append(spearman_corr)
        correlations[i] = {'pearson': pearson_values, 'spearman': spearman_values}

    # Save correlations to file
    correlation_file_path = os.path.join(output_dir, 'correlations.pkl')
    with open(correlation_file_path, 'wb') as f:
        pickle.dump(correlations, f)

    
    return print(f'Correlations saved to {correlation_file_path}')

def plot_correlations(correlation_file_path, save_dir = None):
    
    if save_dir is None:
        save_dir = correlation_file_path
        
    with open(correlation_file_path, 'rb') as f:
        correlations = pickle.load(f)
    
    n_pred_concepts = len(correlations)
    n_unsup_concepts = len(correlations[0]['pearson'])
    
    fig, axs = plt.subplots(n_pred_concepts, figsize=(10, 6 * n_pred_concepts))
    
    if n_pred_concepts == 1:
        axs = [axs]

    for i in range(n_pred_concepts):
        pearson_values = correlations[i]['pearson']
        spearman_values = correlations[i]['spearman']
        
        x_labels = [f'Unsup Concept {j+1}' for j in range(n_unsup_concepts)]
        
        bar_width = 0.35
        x = np.arange(len(x_labels))
        
        axs[i].bar(x - bar_width/2, pearson_values, bar_width, label='Pearson', color='blue')
        axs[i].bar(x + bar_width/2, spearman_values, bar_width, label='Spearman', color='orange')
        
        axs[i].set_title(f'Correlations for Preds Concept {i}')
        axs[i].set_ylabel('Correlation Coefficient')
        axs[i].set_xticks(x)
        axs[i].set_xticklabels(x_labels, rotation=45, ha='right')
        axs[i].grid(axis='y', linestyle='--', alpha=0.7)
        axs[i].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'corr_preds_concept_{i + 1}.png'))
    plt.show()
    plt.close()

############################### RICCARDO  ORTHOGONALITY ########################
#%%
def orth_frob_loss(weight_vectors, sup_unsup_tuple=None, one_vs_all=False):
    """Compute the orthogonal loss for a list of weight vectors.
    
    Parameters:
    weight_vectors (list of torch.Tensor): A list of 1D tensors, where each tensor represents
                                            the weight vector of a linear layer. Each tensor has 
                                            shape (vector_dimension,).
    sup_unsup_tuple (tuple, optional): A tuple (X, Y) to split the weight vectors into two parts.
    one_vs_all (bool, optional): If True, sets the Gram matrix entries to zero for the inner 
                                 products between vectors in the first part specified by sup_unsup_tuple.
    
    Returns:
    torch.Tensor: A scalar tensor representing the Frobenius norm of the difference between 
                  the Gram matrix of the weight vectors and the identity matrix. This loss 
                  encourages the weight vectors to be orthogonal.
    """
    
    if sup_unsup_tuple is None:
        # Concatenate all weight vectors into a single tensor
        concatenated_vectors = torch.cat(weight_vectors, dim=0)
        gram_matrix = torch.matmul(concatenated_vectors, concatenated_vectors.t())
        identity_matrix = torch.eye(gram_matrix.size(0), device=gram_matrix.device)
        frobenius_loss = torch.norm(gram_matrix - identity_matrix, 'fro')
    else:
        X, Y = sup_unsup_tuple
        assert X + Y == len(weight_vectors), "The sum of X and Y must equal the number of weight vectors"

        # Split the weight vectors into two parts
        part_A = torch.cat(weight_vectors[:X], dim=0)
        part_B = torch.cat(weight_vectors[X:X+Y], dim=0)
        
        if one_vs_all:
            # Concatenate all vectors and compute the full Gram matrix
            concatenated_vectors = torch.cat(weight_vectors, dim=0)
            gram_matrix = torch.matmul(concatenated_vectors, concatenated_vectors.t())
            
            # Set the inner products of the first X concepts equal zero
            gram_matrix[:X, :X] = 0
            gram_matrix[X:X+Y, X:X+Y] = 0
            
            # Compute the Frobenius norm of the Gram matrix with zeros in the specified positions
            identity_matrix = torch.eye(gram_matrix.size(0), device=gram_matrix.device)
            frobenius_loss = torch.norm(gram_matrix - identity_matrix, 'fro')
        else:
            # Compute the cross Gram matrix between part_A and part_B
            cross_gram_matrix = torch.matmul(part_A, part_B.t())
            
            # We expect this matrix to be close to zero for orthogonality
            zero_matrix = torch.zeros(cross_gram_matrix.size(), device=cross_gram_matrix.device)
            
            # Compute the Frobenius norm of the cross Gram matrix
            frobenius_loss = torch.norm(cross_gram_matrix - zero_matrix, 'fro')

    return frobenius_loss
#%%
# def mmd_loss(W, target_distribution):
#     # Target distribution ca be created exploiting random orthogonal vectors like this: q, _ = np.linalg.qr(vectors) (Orthonormal upper-triangular)
#     """Compute the Maximal Mean Discrepancy (MMD) loss."""
#     phi_W = np.mean(W, axis=0)  # Feature map for neuron weights
#     phi_V = np.mean(target_distribution, axis=0)  # Feature map for target distribution
#     return np.linalg.norm(phi_W - phi_V)**2

def orth_gram_loss(weight_vectors):
    """
    Compute the orthogonal loss for a list of weight vectors.
    
    Parameters:
    weight_vectors (list of torch.Tensor): A list of 1D tensors, where each tensor represents
                                            the weight vector of a linear layer. Each tensor has 
                                            shape (vector_dimension,).
    
    Returns:
    torch.Tensor: A scalar tensor representing the mean squared difference between the Gram 
                  matrix of the weight vectors and the identity matrix. This loss encourages 
                  the weight vectors to be orthogonal.
    """
    
    # Concatenate all weight vectors into a single tensor
    concatenated_vectors = torch.stack(weight_vectors)
    
    # Compute the Gram matrix
    gram_matrix = torch.matmul(concatenated_vectors, concatenated_vectors.t())
    
    # Compute the mean squared difference from the identity matrix
    identity_matrix = torch.eye(gram_matrix.size(0), device=gram_matrix.device)
    mse_loss = torch.mean((gram_matrix - identity_matrix) ** 2)
    
    return mse_loss

def cosine_similarity_loss(W):
    """Compute the cosine similarity loss for the weight matrix W.
    
    Parameters:
    W (torch.Tensor): A 2D tensor of shape (num_layers, num_neurons) representing the weights of 
                      multiple linear layers. Each row corresponds to the weights of a single 
                      linear layer, mapping `num_neurons` inputs to a single output.
    
    Returns:
    torch.Tensor: A scalar tensor representing the mean of the off-diagonal elements in the 
                  cosine similarity matrix of the weight vectors. This loss encourages the 
                  weight vectors to be orthogonal, as a lower mean similarity indicates more 
                  orthogonality.
    """
    W = W / W.norm(dim=1, keepdim=True)  # Normalize the weight vectors
    cosine_sim_matrix = torch.mm(W, W.t())  # Compute the cosine similarity matrix
    
    # We want the off-diagonal elements to be as close to zero as possible
    batch_size = W.shape[0]
    cosine_sim_matrix.fill_diagonal_(0)  # Zero out the diagonal
    similarity_sum = cosine_sim_matrix.sum() / (batch_size * (batch_size - 1) / 2)
    
# %%

    return similarity_sum
