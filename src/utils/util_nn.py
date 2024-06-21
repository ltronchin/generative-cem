#%%
from tqdm import trange
import torch
import os
import torch.nn as nn
import time
import copy
import numpy as np
import re
import re
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix, roc_auc_score, precision_recall_fscore_support
import torch.nn.functional as F
import pickle
from scipy.stats import pearsonr, spearmanr

# Figure properties.
import seaborn as sns
import matplotlib.pyplot as plt
from umap import UMAP
from sklearn.manifold import TSNE

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
    
    excl_grad_decoder = False
    # Excluding gradient decoder checking if rec_loss is zero as will raise warning for vanishing ones
    if weight_rec_loss == 0.0: 
        if hasattr(model, 'concept_decoder'):
            excl_grad_decoder = True
            
    excl_grad_predictor = False
    # Same with predictor
    if weight_task_loss == 0.0:
        if hasattr(model, 'predictor'):
            excl_grad_predictor = True
           
    if freeze_encoder:
        if hasattr(model, 'concept_encoder'):
            for param in model.concept_encoder.parameters():
                param.requires_grad = False

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion_c = nn.MSELoss()      # Loss dei concetti
    criterion_y = nn.CrossEntropyLoss() # Loss classifcation
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

    # MSE and CORR intiialization
    epoch_mse_dict = {}
    epoch_corr_dict = {}
    
    epochs_no_improve = 0
    best_epoch = 0
    early_stop = False

    since = time.time()
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        # MSE
        epoch_mse_dict[epoch] = {
            'train': {'mean': {}, 'std': {}},
            'val': {'mean': {}, 'std': {}}
        }
        # CORR  
        epoch_corr_dict[epoch] = {
            'train': {'mean': {}, 'std': {}},
            'val': {'mean': {}, 'std': {}}
        }

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if early_stop:
                break
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
            
            # MSE
            running_c_excl_unsup_dist = {}
            running_c_excl_unsup_dist_std = {}
            # CORR
            running_c_excl_unsup_corr = {}
            running_c_excl_unsup_corr_std = {}
            
            counter = 0
            # Iterate over data.
            for sample in tqdm(data_loaders[phase]):

                imgs, c_to_keep, c_excl, labels = sample['image'].to(device), sample['c_to_keep'].to(device), sample['c_excl'].to(device), sample['label'].to(device)
                optimizer.zero_grad()
                counter = counter + 1

                # forward
                with torch.set_grad_enabled(phase == 'train'):
                    # Changing number of output within model types
                    if type(model).__name__ == 'End2End':
                        preds_concepts, unsup_concepts, preds_task, preds_img_tilde, preds_concepts_tilde, linear_weights = model(imgs)
                    elif type(model).__name__ == 'IndependentMLP':
                        preds_concepts, unsup_concepts, preds_task = model(imgs)
                        # Creating missing variables by casting them to original value for conserving loss calculation
                        preds_img_tilde = imgs
                        preds_concepts_tilde = preds_concepts
                    elif type(model).__name__ == 'Encoder':
                        preds_concepts, unsup_concepts, linear_weights = model(imgs)
                        # Creating missing variables for losses by casting them to original value for conserving loss calculation
                        preds_task = torch.zeros((imgs.size(0), 2), dtype=torch.float32, device=device)
                        preds_img_tilde = imgs
                        preds_concepts_tilde = preds_concepts
                    else:
                        raise ValueError('Model type not found!')
                    
                    # The following variables is not utilized                       
                    # _, preds = torch.max(preds_task, 1)

                    loss = torch.tensor(0.0).to(device)

                    concept_loss = criterion_c(preds_concepts, c_to_keep)
                    loss += (weight_concept_loss * concept_loss)
                    
                    task_loss = criterion_y(preds_task, labels)
                    loss += (weight_task_loss * task_loss)

                    rec_loss = criterion_rec(preds_img_tilde, imgs)
                    loss += (weight_rec_loss * rec_loss)

                    lat_loss = criterion_lat(preds_concepts_tilde, c_to_keep)
                    loss += (weight_lat_loss * lat_loss)
                    
                    # orth_loss = cosine_similarity_loss(linear_weights)
                    if unsup_concepts is not None: 
                        orth_loss = orth_frob_loss(linear_weights, sup_unsup_tuple=(preds_concepts.size(1), unsup_concepts.size(1)), one_vs_all=True)
                        # orth_loss = orth_gram_loss(linear_weights)
                        loss += (weight_orth_loss * orth_loss)
                    else:
                        orth_loss =  torch.tensor(0.0).to(device)
                    
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        
                        # Gradient boundary check
                        grad_threshold = 10  # Example threshold value
                        for name, param in model.named_parameters():
                            # exclude decoder gradients if no loss on decoder, otherwise always zeros and warnings.
                            if excl_grad_decoder and 'concept_decoder' in name:
                               continue
                            if excl_grad_predictor and 'predictor' in name:
                               continue
                            if 'bias' not in name:  #  don't care if the grads of the bias is low, as it affect only starting point (that can be 0)
                                if param.grad is not None:
                                    grad_norm = param.grad.norm()
                                    if grad_norm > grad_threshold:
                                        print(f"Warning: Gradient norm for {name}, SHAPE: {param.shape} exceeds threshold: {grad_norm:.8f}")
                                    elif grad_norm < 1e-6:  # Example lower boundary for vanishing gradients
                                        print(f"Warning: Gradient norm for {name}, SHAPE: {param.shape} is too small: {grad_norm:.8f}")
                                        
                # No grad needed for measure distances between superv Cs and unsuperv Cs                   
                with torch.no_grad():     
                    if unsup_concepts is not None:
                    # Calculate pairwise MSE (average over batch dimension) on CORR
                        for i in range(c_excl.size(1)):
                                for j in range(unsup_concepts.size(1)):
                                    if (i, j) not in running_c_excl_unsup_dist:
                                        # MSE
                                        running_c_excl_unsup_dist[(i, j)] = []
                                        running_c_excl_unsup_dist_std[(i, j)] = []
                                        # CORR
                                        running_c_excl_unsup_corr[(i, j)] = []
                                        running_c_excl_unsup_corr_std[(i, j)] = []

                                    # MSE
                                    mse_batch = F.mse_loss(c_excl[:, i], unsup_concepts[:, j], reduction='none').cpu().detach().numpy()
                                    mse_mean = np.mean(mse_batch, axis = 0)
                                    mse_std = np.std(mse_batch, axis=0)
                                 
                                    running_c_excl_unsup_dist[(i, j)].append(mse_mean)
                                    running_c_excl_unsup_dist_std[(i, j)].append(mse_std)     
                                    
                                    # CORR
                                    c_excl_i = c_excl[:, i].cpu().detach().numpy()
                                    unsup_concepts_j = unsup_concepts[:, j].cpu().detach().numpy()

                                    corr_batch, _ = pearsonr(c_excl_i, unsup_concepts_j)
                                    running_c_excl_unsup_corr[(i, j)].append(corr_batch)                           

                # Statistics
                running_task_loss += task_loss.item()
                running_concept_loss += concept_loss.item()
                running_rec_loss += rec_loss.item()
                running_lat_loss += lat_loss.item()
                running_orth_loss += orth_loss.item()
                running_loss += loss.item()
            
            # print(f'PREDS_C: {preds_concepts}\nCONCEPTS: {concepts}')
            # print(f'PREDS_T: {preds_task}\nLABELS: {labels}')

            epoch_task_loss = running_task_loss / len(data_loaders[phase])
            epoch_concept_loss = running_concept_loss / len(data_loaders[phase])
            epoch_rec_loss = running_rec_loss / len(data_loaders[phase])
            epoch_lat_loss = running_lat_loss / len(data_loaders[phase])
            epoch_orth_loss = running_orth_loss / len(data_loaders[phase])
            epoch_loss = running_loss / len(data_loaders[phase])
            
            if unsup_concepts is not None:
                # Average over total number of batches
                for key in running_c_excl_unsup_dist:
                    # MSE
                    # print(f'KEY:  {key}')
                    running_c_excl_unsup_dist[key] = np.mean(running_c_excl_unsup_dist[key])
                    # print(f'RUN_SUP_UNSUP_MEAN_EPOCH: {running_c_excl_unsup_dist[key]}')
                    
                    running_c_excl_unsup_dist_std[key] = np.mean(running_c_excl_unsup_dist_std[key])
                    # print(f'RUN_SUP_UNSUP_STD_EPOCH: {running_c_excl_unsup_dist_std[key]}')
                    
                    running_c_excl_unsup_corr[key] = np.mean(running_c_excl_unsup_corr[key])
                    running_c_excl_unsup_corr_std[key] = np.std(running_c_excl_unsup_corr[key])
                
                # MSE   
                epoch_mse_dict[epoch][phase]['mean'] = running_c_excl_unsup_dist
                epoch_mse_dict[epoch][phase]['std'] = running_c_excl_unsup_dist_std
                # CORR
                epoch_corr_dict[epoch][phase]['mean'] = running_c_excl_unsup_corr
                epoch_corr_dict[epoch][phase]['std'] = running_c_excl_unsup_corr_std

                                
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
                            print(f'\nEarly Stopping! Total epochs: {epoch}')
                            early_stop = True
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

    return model, history, epoch_mse_dict, epoch_corr_dict
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
            inputs, concepts, labels = sample['image'].to(device), sample['c_to_keep'].to(device), sample['label'].to(device)
            all_labels.extend(labels.cpu().numpy())
            all_concepts.extend(concepts.cpu().numpy())

            # Prediction
            if type(model).__name__ == 'End2End':
                preds_concepts, unsup_concepts, preds_task, preds_rec, preds_concepts_tilde, _ = model(inputs)
            elif type(model).__name__ == 'IndependentMLP':
                preds_concepts, unsup_concepts, preds_task = model(inputs)
                # Creating missing variables by casting them to original value for conserving loss calculation
                preds_rec = inputs
                preds_concepts_tilde = preds_concepts
            elif type(model).__name__ == 'Encoder':
                preds_concepts, unsup_concepts, _ = model(inputs)
                # Creating missing variables for losses by casting them to original value for conserving loss calculation
                preds_task = torch.zeros((inputs.size(0), 2), dtype=torch.float32, device=device)
                preds_rec = inputs
                preds_concepts_tilde = preds_concepts
            else:
                raise ValueError('Model type not found!')            
    
            _, preds = torch.max(preds_task, 1)

            all_preds.extend(preds.cpu().numpy())
            all_preds_concepts.extend(preds_concepts.cpu().numpy())
            all_rec.append(F.mse_loss(preds_rec, inputs).item())
            all_lat.append(F.mse_loss(preds_concepts_tilde, concepts).item())

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
            inputs, _, _ = sample['image'].to(device), sample['c_to_keep'].to(device), sample['label'].to(device)
            
            if type(model).__name__ == 'End2End':
                preds_concepts, _, _, _, _, _ = model(inputs)
                
            elif type(model).__name__ == 'IndependentMLP':
                preds_concepts, _, _ = model(inputs)
                
            elif type(model).__name__ == 'Encoder':
                preds_concepts, _, _ = model(inputs)

            else:
                raise ValueError('Model type not found!')      

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

def plot_training(history, plot_training_dir, loss_names, plot_name_loss='Loss', plot_name_acc='Acc', precision = 0.001):

    # Training results Loss function
    plt.figure()
    for c in loss_names:
        if abs(history[c].iloc[-1] - history[c].iloc[0]) >= precision:
            plt.plot(history[c], label=c)
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Losses')

    # Delete white space
    plt.tight_layout()
    plt.savefig(os.path.join(plot_training_dir, f"{plot_name_loss}.png"),  dpi=400, format='png')
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
        plt.savefig(os.path.join(plot_training_dir, f"{plot_name_acc}.png"),  dpi=400, format='png')
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
    x_out.save(os.path.join(plot_training_dir, "img_rec.png"), resolution=400)
    
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
    
def plot_c_excl_unsup_mse_epoch(epoch_mse_errors_dict, concepts_size, unsup_concepts_size, save_dir, data_type = 'corr'):
    num_epochs = len(epoch_mse_errors_dict)
    
    fig, axes = plt.subplots(1, concepts_size, figsize=(15, 5))  # Create subplots
    for i in range(concepts_size):
        ax = axes[i]
        
        for j in range(unsup_concepts_size):
            train_means = [epoch_mse_errors_dict[epoch]['train']['mean'].get((i, j), 0 ) for epoch in range(num_epochs)]
            val_means = [epoch_mse_errors_dict[epoch]['val']['mean'].get((i, j), 0) for epoch in range(num_epochs)]
            
            train_stds = [epoch_mse_errors_dict[epoch]['train']['std'].get((i, j),0) for epoch in range(num_epochs)]
            val_stds = [epoch_mse_errors_dict[epoch]['val']['std'].get((i, j), 0) for epoch in range(num_epochs)]
            
            # Plot mean values with error bars (standard deviation)
            ax.errorbar(range(1, num_epochs + 1), train_means, yerr=train_stds, label=f'unsup C{j + 1} Train', linestyle='-', color=f'C{j}')
            ax.errorbar(range(1, num_epochs + 1), val_means, yerr=val_stds, label=f'unsup C{j + 1} Val', linestyle='--', color=f'C{j}')
        
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Pearson Correlation' if data_type == 'corr' else data_type)
        ax.set_title(f'Excluded concept {i + 1}')
        
            
        if unsup_concepts_size <= 3:
            ax.legend(title='Legend:', loc='best', fontsize='small')
        
        # Additional text box for clarity
        if unsup_concepts_size > 3:
            ax.text(0.5, 0.92, 'Continuous line: Train\n-- line: Validation', horizontalalignment='center', 
                    verticalalignment='center', fontsize=9,
                    transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.5))
    
    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{data_type}_epoch_concepts_vs_unsup.png'))
    plt.show()
    plt.close()

#################################################### correlations and distances post taining ############################
#%%
def calculate_and_save_distances(model, data_loaders, device, output_dir):
    model.eval()  # Set model to evaluation mode
    
    distance_dict = {}

    for phase in ['train', 'val', 'test']:
        count = 0
        for idx, sample in tqdm(enumerate(data_loaders[phase]), desc=f'Calculating distances for {phase}'):
            if count >= 1000:
                break
            
            imgs, excl_concepts = sample['image'].to(device), sample['c_excl'].to(device)

            with torch.no_grad():
                if type(model).__name__ == 'End2End':
                    preds_concepts, unsup_concepts, _, _, _, _ = model(imgs)
                elif type(model).__name__ == 'IndependentMLP':
                    preds_concepts, unsup_concepts, _ = model(imgs)
                elif type(model).__name__ == 'Encoder':
                    preds_concepts, unsup_concepts,_ = model(imgs)

                else:
                    raise ValueError('Model type not found!')
                # print(f'SUP_VS_UNSUP_CONCEPT:\n{preds_concepts}\n{unsup_concepts}')

                sample_distances = np.zeros((excl_concepts.size(1), unsup_concepts.size(1)))

                for i in range(excl_concepts.size(1)):
                    for j in range(unsup_concepts.size(1)):
                        # Batch size = 1 so reduction is none
                        mse_value = F.mse_loss(excl_concepts[:, i], unsup_concepts[:, j], reduction='none').item()
                        sample_distances[i, j] = mse_value

                # print(f'SUP_CONCEPTS_VS_UNSUP_CONC_1:\n{sample_distances[0,1]}, {sample_distances[1,1]}, {sample_distances[2,1]}')

            distance_dict[(phase, idx)] = sample_distances
            
            count += 1
        # print(f'DIST DICT: -> {np.shape(distance_dict[(phase, idx-1)])}')
    # print(distance_dict)
    
    
    # Save distances to file
    distance_file_path = os.path.join(output_dir, 'distances.pkl')
    with open(distance_file_path, 'wb') as f:
        pickle.dump(distance_dict, f)


    return print(f'Distances saved to {distance_file_path}')

def plot_mse_values(distance_file_path, save_dir=None):
    
    if save_dir is None:
        save_dir = os.path.dirname(distance_file_path)
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
        print(f'Mean over 1000 samples of distances between concept {i+1} and the {n_unsup_concepts} Unsupervised:\n{mean_mse_values}')
        
        x_labels = [f'Unsup C{j+1}' for j in range(n_unsup_concepts)]
        
        bar_width = 0.35
        x = np.arange(len(x_labels))
        
        axs[i].bar(x, mean_mse_values, bar_width, label=f'Preds Concept {i}', color='blue')
        
        axs[i].set_title(f'MSE Values for Predicted Concept {i+1}')
        axs[i].set_ylabel('Mean MSE')
        axs[i].set_xticks(x)
        axs[i].set_xticklabels(x_labels, rotation=45, ha='right')
        axs[i].grid(axis='y', linestyle='--', alpha=0.7)
        axs[i].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'mse_pred_concepts_vs_unsup_concepts.png'))
    plt.show()
    plt.close()
    
    return print(f'Plot saved to {save_dir}')
    
    
def calculate_and_save_correlations(model, data_loaders, device, output_dir, sup_unsup=False):
    model.eval()  # Set model to evaluation mode

    preds_concepts_list = []
    unsup_concepts_list = []
    excl_concepts_list = []

    for phase in ['train', 'val', 'test']:
        count = 0
        for idx, sample in tqdm(enumerate(data_loaders[phase]), desc=f'Extracting concepts for {phase}'):
            if count >= 1000:
                break
            
            imgs, excl_concepts = sample['image'].to(device), sample['c_excl'].to(device)

            with torch.no_grad():
                if type(model).__name__ == 'End2End':
                    preds_concepts, unsup_concepts, _, _, _, _ = model(imgs)
                elif type(model).__name__ == 'IndependentMLP':
                    preds_concepts, unsup_concepts, _ = model(imgs)
                elif type(model).__name__ == 'Encoder':
                    preds_concepts, unsup_concepts,_ = model(imgs)
                else:
                    raise ValueError('Model type not found!')

                preds_concepts_list.append(preds_concepts.cpu().numpy())
                unsup_concepts_list.append(unsup_concepts.cpu().numpy())
                excl_concepts_list.append(excl_concepts)
            
            count += 1

    preds_concepts_array = np.concatenate(preds_concepts_list, axis=0)
    unsup_concepts_array = np.concatenate(unsup_concepts_list, axis=0)
    excl_concepts_array = np.concatenate(excl_concepts_list, axis=0)

    # If varaible true change to perform correlation between sup and unsup
    if sup_unsup:
        excl_concepts_array = preds_concepts_array

    n_pred_concepts = preds_concepts_array.shape[1]
    n_unsup_concepts = unsup_concepts_array.shape[1]
    n_excl = excl_concepts_array.shape[1]

    correlations = {}
        
    for i in range(n_excl):
        pearson_values = []
        spearman_values = []
        for j in range(n_unsup_concepts):
            # Pearson correlation
            pearson_corr, _ = pearsonr(excl_concepts_array[:, i], unsup_concepts_array[:, j])
            # Spearman correlation
            spearman_corr, _ = spearmanr(excl_concepts_array[:, i], unsup_concepts_array[:, j])
            pearson_values.append(pearson_corr)
            spearman_values.append(spearman_corr)
        correlations[i] = {'pearson': pearson_values, 'spearman': spearman_values}

    # Save correlations to file
    if sup_unsup:
        correlation_file_path  = os.path.join(output_dir, 'correlations_sup_unsup.pkl')
    else:
        correlation_file_path = os.path.join(output_dir, 'correlations_excl_unsup.pkl')
    with open(correlation_file_path, 'wb') as f:
        pickle.dump(correlations, f)

    
    return print(f'Correlations saved to {correlation_file_path}')

def plot_correlations(correlation_file_path, save_dir = None, title = None, name = None):
    
    if save_dir is None:
        save_dir = os.path.dirname(correlation_file_path)
        
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
        
        x_labels = [f'Unsup C{j+1}' for j in range(n_unsup_concepts)]
        
        bar_width = 0.35
        x = np.arange(len(x_labels))
        
        axs[i].bar(x - bar_width/2, pearson_values, bar_width, label='Pearson', color='blue')
        axs[i].bar(x + bar_width/2, spearman_values, bar_width, label='Spearman', color='orange')
        if title == None:
            axs[i].set_title(f'Correlations for Concept {i+1}')
        else: 
            axs[i].set_title(f'{title} for Concept {i+1}')
        axs[i].set_ylabel('Correlation Coefficient')
        axs[i].set_xticks(x)
        axs[i].set_xticklabels(x_labels, rotation=45, ha='right')
        axs[i].grid(axis='y', linestyle='--', alpha=0.7)
        axs[i].legend()

    plt.tight_layout()
    if name == None:
        name = re.search(r'correlations_(.*?)\.pkl', correlation_file_path)
        # print(name.group(1))
        plt.savefig(os.path.join(save_dir, f"corr_{name.group(1)}.png"))
    else:
        plt.savefig(os.path.join(save_dir, f'{name}.png'))
    plt.show()
    plt.close()
    
#############################################################################################      
############################### RICCARDO  ORTHOGONALITY ########################
#############################################################################################
#%%
def orth_frob_loss(weight_vectors, sup_unsup_tuple=None, one_vs_all=False):
    """Compute the orthogonal loss for a list of weight vectors.
    
    Parameters:
    weight_vectors (list of torch.Tensor): A list of 1D tensors, where each tensor represents
                                            the weight vector of a linear layer. Each tensor has 
                                            shape (vector_dimension,).
    sup_unsup_tuple (tuple, optional): A tuple (X, Y) to split the weight vectors into two parts.
    one_vs_all (bool, optional): If True, sets the Gram matrix entries to zero for the inner 
                                 products between vectors in the first part (e.g. supervised) specified by sup_unsup_tuple.
    
    Returns:
    torch.Tensor: A scalar tensor representing the Frobenius norm of the difference between 
                  the Gram matrix of the weight vectors and the identity matrix. This loss 
                  encourages the weight vectors to be orthogonal.
    """
    weight_vectors = torch.stack(weight_vectors, dim=0).squeeze(1)
    # print(weight_vectors.shape)
    if sup_unsup_tuple is None:
        # Compute the Gram matrix for each batch
        gram_matrix = torch.matmul(weight_vectors, weight_vectors.t())
        # print(gram_matrix.shape)
        identity_matrix = torch.eye(gram_matrix.size(0), device=weight_vectors.device)
        frobenius_loss = torch.norm(gram_matrix - identity_matrix, 'fro')
    else:
        X, Y = sup_unsup_tuple
    if Y is None:
        return torch.tensor(0.0, device=weight_vectors.device)

    else:
        X, Y = int(X), int(Y)
        assert X + Y == weight_vectors.size(0), "The sum of X and Y must equal the number of weight vectors"

    # Split the weight vectors into two parts
    part_A = weight_vectors[:X]
    part_B = weight_vectors[X:X+Y]
    # print('Part A and B',part_A.shape, part_B.shape)
    
    if one_vs_all:
        # Compute the full Gram matrix for each batch
        gram_matrix = torch.matmul(weight_vectors, weight_vectors.t())
        
        # Set the inner products of the first X concepts to zero
        gram_matrix[:X, :X] = 0
        gram_matrix[X:X+Y, X:X+Y] = 0
        # print(gram_matrix.shape)
        
        # Compute the Frobenius norm of the Gram matrix with zeros in the specified positions
        identity_matrix = torch.eye(weight_vectors.size(0), device=weight_vectors.device)
        frobenius_loss = torch.norm(gram_matrix - identity_matrix, 'fro')
    else:
        # Compute the cross Gram matrix between part_A and part_B for each batch
        cross_gram_matrix = torch.matmul(part_A, part_B.transpose(1, 2))
        print(cross_gram_matrix.shape)
        
        # We expect this matrix to be close to zero for orthogonality
        zero_matrix = torch.zeros(cross_gram_matrix.size(), device=cross_gram_matrix.device)
        
        # Compute the Frobenius norm of the cross Gram matrix
        frobenius_loss = torch.norm(cross_gram_matrix - zero_matrix, 'fro')

    # Return the mean loss across all batches
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
    
    
    return similarity_sum
#################################################################################################################
################################################################################################################# UMAP AND TSNE      
#################################################################################################################

#%%
def run_tsne(model, data_loader, savedir=None, device='cpu', perplexity=30, n_iter=1000, n_neighbors = 15, min_dist = 0.1):
    model.to(device)
    model.eval()
    
    for phase in ['train']: #, 'val', 'test']:
        sup_concept_list = []
        unsup_concept_list = []
        label_list = []
        weights_list = []
        with torch.no_grad():
            for idx, sample in tqdm(enumerate(data_loader[phase]), desc=f'Extracting concepts from {phase}'):
                
                imgs, concepts = sample['image'].to(device), sample['c_to_keep'].to(device)
                # Get dimensions
                # if idx == 0:                                       
                    # n_sup = preds_concepts.size(1)
                    # n_unsup = unsup_concepts.size(1)
                    # n_emb = linear_weights[0].shape[1]
                
                if idx >= 1000:
                
                    break
                if type(model).__name__ == 'End2End':
                    preds_concepts, unsup_concepts, preds_task, preds_img_tilde, preds_concepts_tilde, linear_weights = model(imgs)
                elif type(model).__name__ == 'IndependentMLP':
                    preds_concepts, unsup_concepts, preds_task = model(imgs)
                elif type(model).__name__ == 'Encoder':
                    preds_concepts, unsup_concepts, linear_weights = model(imgs)
                else:
                    raise ValueError('Model type not found!')
                
                # Unique label for concepts
                label_list.append(np.argmax(preds_task.numpy()))
                # Supervised and unsupervised concepts 
                sup_concept_list.append(preds_concepts.cpu().numpy())   # list of elemts with shape (1, n_sup)
                unsup_concept_list.append(unsup_concepts.cpu().numpy())  # list of elemts with shape (1, n_unsup)
                
                # Weights
                weights = np.vstack([weight.cpu().numpy() for weight in linear_weights]).reshape(-1)   # ((n_sup + n_unsup) * n_emb,)
                weights_list.append(weights)

        # Stack and concatenate concepts
        sup_concept_list = np.vstack(sup_concept_list)  # (idx, n_sup)
        unsup_concept_list = np.vstack(unsup_concept_list)  # (idx, n_unsup)
        combined_concepts = np.hstack((sup_concept_list, unsup_concept_list))   # (idx, n_sup + n_unsup)

        # Stack and concatenate weights
        combined_weights = np.vstack(weights_list)    # (idx, n_emb * (n_sup + n_unsup))
           
        # Print dimensions
        print(f"\nPhase: {phase}")
        print(f"sup_concept_list shape: {sup_concept_list.shape}")
        print(f"unsup_concept_list shape: {unsup_concept_list.shape}")
        print(f"combined_concepts shape: {combined_concepts.shape}")
        print(f"combined weights shape: {combined_weights.shape}")


        # Perform t-SNE for concepts
        tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=42)
        concept_tsne = tsne.fit_transform(combined_concepts)
        
        # # Perform t-SNE for weights
        # tsne_weights = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=42)
        # c_weights_tsne = tsne_weights.fit_transform(combined_weights)
        
        # Perform UMAP for weights
        umap_weights = UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
        c_weights_umap = umap_weights.fit_transform(combined_weights)
        
        plt.subplot(1, 2, 1)
        for i in range(len(label_list)):
            if label_list[i] == 0:
                color = 'blue'  # Use blue for label 0
            else:
                color = 'red'   # Use red for label 1
            
            plt.scatter(concept_tsne[i, 0], concept_tsne[i, 1], color=color)
        
        plt.title(f't-SNE of Concepts in {phase}')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.legend()
        
        # Plot t-SNE for weights
        plt.subplot(1, 2, 2)
        for i in range(len(label_list)):
            if label_list[i] == 0:
                color = 'blue'  # Use blue for label 0
            else:
                color = 'red'   # Use red for label 1
            
            # plt.scatter(c_weights_tsne[i, 0], c_weights_tsne[i, 1], color=color)
            plt.scatter(c_weights_umap[i, 0], c_weights_umap[i, 1], color=color)
        
        plt.title(f'UMAP of Weights in {phase}')
        plt.xlabel('UMAP Component 1')
        plt.ylabel('UMAP Component 2')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
        
                # Save or show plot
        if savedir:
            if not os.path.exists(savedir):
                os.makedirs(savedir)
            plt.savefig(os.path.join(savedir, f'{phase}_tsne_full.png'))
            print(f'T-SNE saved at {savedir}/{phase}_tsne_full.png')

        
#%%

#############################################################################
#############################################################################
#############################################################################
#############################################################################
def run_tsne2(model, data_loader, savedir=None, device='cpu', perplexity=30, n_iter=1000, n_neighbors=30, min_dist=0.1):
    model.to(device)
    model.eval()
    
    for phase in ['train']: # , 'val', 'test']:
        sup_concept_list = []
        unsup_concept_list = []
        sup_weights_list = []
        unsup_weights_list = []
        label_list = []
        
        with torch.no_grad():
            for idx, sample in tqdm(enumerate(data_loader[phase]), desc=f'Extracting concepts from {phase}'):
                
                imgs, concepts = sample['image'].to(device), sample['c_to_keep'].to(device)
                    
                if idx >= 1000:
                    break
                
                if type(model).__name__ == 'End2End':
                    preds_concepts, unsup_concepts, preds_task, preds_img_tilde, preds_concepts_tilde, linear_weights = model(imgs)
                elif type(model).__name__ == 'IndependentMLP':
                    preds_concepts, unsup_concepts, preds_task = model(imgs)
                    # Creating missing variables by casting them to original value for conserving loss calculation
                    preds_img_tilde = imgs
                    preds_concepts_tilde = preds_concepts
                elif type(model).__name__ == 'Encoder':
                    preds_concepts, unsup_concepts, linear_weights = model(imgs)
                    # Creating missing variables for losses by casting them to original value for conserving loss calculation
                    preds_task = torch.zeros((imgs.size(0), 2), dtype=torch.float32, device=device)
                    preds_img_tilde = imgs
                    preds_concepts_tilde = preds_concepts
                else:
                    raise ValueError('Model type not found!')
                
                # Unique label for concepts
                label_list.append(np.argmax(preds_task.numpy()))
                
                # Supervised concepts and weights
                sup_concept_list.append(preds_concepts.cpu().numpy())
                preds_concepts_weights = np.vstack([weight.cpu().numpy() for weight in linear_weights[:preds_concepts.size(1)]]).reshape(-1)   # 1D (n_sup*16,)
                sup_weights_list.append(preds_concepts_weights)      # len(list) = idx  list[x].shape = (n_sup * 16,)
                
                # Unsupervised concepts and weights
                unsup_concept_list.append(unsup_concepts.cpu().numpy())             
                unsup_concepts_weights = np.vstack([weight.cpu().numpy() for weight in linear_weights[preds_concepts.size(1):]]).reshape(-1)    # 1D = (n_unsup*16,)
                unsup_weights_list.append(unsup_concepts_weights)      # len(list) = idx  -> list[x].shape = (n_unsup * 16,)
        
        n_sup = preds_concepts.size(1)
        n_unsup = unsup_concepts.size(1)
        if n_sup != n_unsup:
            raise ValueError('Number of supervised and unsupervised concepts not equal!\n\
                                This TSNE overimpose unsupervised and supervised spaces as different samples')
        # Stack supervised and unsupervised concepts
        sup_concept_list = np.vstack(sup_concept_list) 
        unsup_concept_list = np.vstack(unsup_concept_list) 
        combined_concepts = np.vstack((sup_concept_list, unsup_concept_list))
        
        # Stack supervised and unsuperised weights
        sup_weights_list = np.vstack(sup_weights_list)        # (idx, n_sup * 16)
        unsup_weights_list = np.vstack(unsup_weights_list)     #  (idx, n_unsup * 16)
        combined_weights = np.vstack((sup_weights_list, unsup_weights_list))    # (2*idx, n_sup * 16)
        
        # Prepare labels and shapes
        # Labels are euqal for both supervised and unsupervised
        label_list = np.concatenate((np.vstack(label_list), np.vstack(label_list))).reshape(-1)    # (2*idx, )
        combined_shapes =  np.concatenate((np.zeros(idx), np.ones(idx)))      # (2*idx,) -> first idx are zeros, supervised. Other idx are 1, unsup.

        # Print dimensions
        print(f"\nPhase: {phase}")
        print(f"sup_concept_list shape: {sup_concept_list.shape}")
        print(f"unsup_concept_list shape: {unsup_concept_list.shape}")
        print(f"combined_concepts shape: {combined_concepts.shape}")
        print(f"combined_labels unique: {np.unique(label_list)}")
        print(f"combined_weights shape: {combined_weights.shape}")

        # Perform t-SNE for concepts
        tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=42)
        combined_tsne = tsne.fit_transform(combined_concepts)
        
        # Perform t-SNE for weights
        tsne_weights = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=42)
        combined_weight_tsne = tsne_weights.fit_transform(combined_weights)        
        
        # UMAP for concepts
        umap = UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
        combined_umap = umap.fit_transform(combined_concepts)
        
        # UMAP for weights
        umap_weights = UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
        combined_weight_umap = umap_weights.fit_transform(combined_weights)
        
        markers = ['o', 'v']    # Circle for supervised, triangle for unsupervised
        colors = ['blue', 'red']        # Blue for label 0, red for label 1    
        
        ####################################################################### Plotting TSNE
        plt.figure(figsize=(16, 8))
        
        # Plot t-SNE for concepts
        plt.subplot(1, 2, 1)       
        for i in range(2*idx):
            # first 1000 samples superv, other 1000 unsupervis
            marker = markers[int(i >= idx)]  
            color = colors[label_list[i]]
            plt.scatter(combined_tsne[i, 0], combined_tsne[i, 1], marker=marker, color=color, label=f'Pred class {0}')
        
        plt.title(f't-SNE of Concepts in {phase}')
        plt.ylabel('t-SNE Component 2')
        
        # Create legend for concepts
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label='Supervised', markerfacecolor='black', markersize=10),
                           plt.Line2D([0], [0], marker='v', color='w', label='Unsupervised', markerfacecolor='black', markersize=10)]
        plt.legend(handles=legend_elements, title='Concept Type')
        
        # Plot t-SNE for weights
        plt.subplot(1, 2, 2)      
        
        for i in range(2*idx):
            # first 1000 samples superv, other 1000 unsupervis
            marker = markers[int(i >= idx)]                  
            color = colors[label_list[i]]
            plt.scatter(combined_weight_tsne[i, 0], combined_weight_tsne[i, 1], marker=marker, color=color, label=f'Pred class {0}')
        
        plt.title(f't-SNE of Weights in {phase}')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        
        # Create legend for weights
        legend_elements_weights = [plt.Line2D([0], [0], marker='o', color='w', label='Supervised', markerfacecolor='black', markersize=10),
                                   plt.Line2D([0], [0], marker='v', color='w', label='Unsupervised', markerfacecolor='black', markersize=10)]
        plt.legend(handles=legend_elements_weights, title='Concept Type')
        
        plt.tight_layout()
        
        # Save or show plot
        if savedir:
            if not os.path.exists(savedir):
                os.makedirs(savedir)
            plt.savefig(os.path.join(savedir, f'{phase}_tsne_overimpose.png'))
            print(f'T-SNE saved at {savedir}/{phase}_tsne_overimpose.png')
        
        plt.show()
        
    ############################################# plotting UMAP
        
        # Plotting
    plt.figure(figsize=(16, 8))
    
    # Plot UMAP for concepts
    plt.subplot(1, 2, 1)
    for i in range(2 * idx):
        marker = markers[int(i >= idx)]
        color = colors[label_list[i]]
        plt.scatter(combined_umap[i, 0], combined_umap[i, 1], marker=marker, color=color, label=f'Pred class {0}')
    
    plt.title(f'UMAP of Concepts in {phase}')
    plt.ylabel('UMAP Component 2')
    
    # Create legend for concepts
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', label='Supervised', markerfacecolor='black', markersize=10),
        plt.Line2D([0], [0], marker='v', color='w', label='Unsupervised', markerfacecolor='black', markersize=10)
    ]
    plt.legend(handles=legend_elements, title='Concept Type')
    
    # Plot UMAP for weights
    plt.subplot(1, 2, 2)
    for i in range(2 * idx):
        marker = markers[int(i >= idx)]
        color = colors[label_list[i]]
        plt.scatter(combined_weight_umap[i, 0], combined_weight_umap[i, 1], marker=marker, color=color, label=f'Pred class {0}')
    
    plt.title(f'UMAP of Weights in {phase}')
    plt.xlabel('UMAP Component 1')
    plt.ylabel('UMAP Component 2')
    
    # Create legend for weights
    legend_elements_weights = [
        plt.Line2D([0], [0], marker='o', color='w', label='Supervised', markerfacecolor='black', markersize=10),
        plt.Line2D([0], [0], marker='v', color='w', label='Unsupervised', markerfacecolor='black', markersize=10)
    ]
    plt.legend(handles=legend_elements_weights, title='Concept Type')
    
    plt.tight_layout()
    
    # Save or show plot
    if savedir:
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        plt.savefig(os.path.join(savedir, f'{phase}_umap_overimpose.png'))
        print(f'UMAP saved at {savedir}/{phase}_umap_overimpose.png')
    
    plt.show()

#############################################################################
#############################################################################
#############################################################################
#############################################################################
#%%
def run_tsne3(model, data_loader, savedir=None, device='cpu', perplexity=30, n_iter=1000):
    model.to(device)
    model.eval()
    
    for phase in ['train']: # , 'val', 'test']:
        concept_list = []
        sup_weights_list = []
        unsup_weights_list = []
        label_list = []
        
        with torch.no_grad():
            for idx, sample in tqdm(enumerate(data_loader[phase]), desc=f'Extracting concepts from {phase}'):
                
                imgs, concepts = sample['image'].to(device), sample['c_to_keep'].to(device)
                    
                if idx >= 1000:
                    break
                
                if type(model).__name__ == 'End2End':
                    preds_concepts, unsup_concepts, preds_task, preds_img_tilde, preds_concepts_tilde, linear_weights = model(imgs)
                elif type(model).__name__ == 'IndependentMLP':
                    preds_concepts, unsup_concepts, preds_task = model(imgs)
                    # Creating missing variables by casting them to original value for conserving loss calculation
                    preds_img_tilde = imgs
                    preds_concepts_tilde = preds_concepts
                elif type(model).__name__ == 'Encoder':
                    preds_concepts, unsup_concepts, linear_weights = model(imgs)
                    # Creating missing variables for losses by casting them to original value for conserving loss calculation
                    preds_task = torch.zeros((imgs.size(0), 2), dtype=torch.float32, device=device)
                    preds_img_tilde = imgs
                    preds_concepts_tilde = preds_concepts
                else:
                    raise ValueError('Model type not found!')
                
                # Unique label for concepts
                label_list.append(np.argmax(preds_task.numpy()))
                
                # Uniqe 1D concept list
                concept_list.append(torch.cat((preds_concepts, unsup_concepts), dim=1).cpu().numpy()) 
                
                # Supervised weights
                preds_concepts_weights = np.vstack([weight.cpu().numpy() for weight in linear_weights[:preds_concepts.size(1)]])   # 1D (n_sup*16,)
                sup_weights_list.append(preds_concepts_weights)      # len(list) = idx  list[x].shape = (n_sup * 16,)
                
                # Unsupervise weights            
                unsup_concepts_weights = np.vstack([weight.cpu().numpy() for weight in linear_weights[preds_concepts.size(1):]])   # 1D = (n_unsup*16,)
                unsup_weights_list.append(unsup_concepts_weights)      # len(list) = idx  -> list[x].shape = (n_unsup * 16,)

        n_sup = preds_concepts.size(1)
        n_unsup = unsup_concepts.size(1)
        # Stack supervised and unsupervised concepts
        concept_list = np.vstack(concept_list).reshape(-1)
        
        # Stack supervised and unsupervised weights
        sup_weights_list = np.vstack(sup_weights_list)      # (n_sample * n_sup, 16)
        unsup_weights_list = np.vstack(unsup_weights_list)     
        combined_weights = np.vstack((sup_weights_list, unsup_weights_list))    # (n_sample * ( n_sup + n_unsup), 16)
        
        # Prepare labels and shapes
        # Labels are euqal for both supervised and unsupervised
        label_list = np.repeat(np.vstack(label_list), (n_sup+n_unsup)).reshape(-1)    
        combined_shapes =  np.concatenate((np.zeros(n_sup*idx), np.ones(n_unsup*idx)))  

        # Print dimensions
        print(f"\nPhase: {phase}")
        print(f"combined_concepts shape: {concept_list.shape}")
        print(f"combined_shapes length: {len(combined_shapes)}")
        print(f"combined_labels length: {label_list.shape}")
        print(f"combined_weights shape: {combined_weights.shape}")
        print(f"combined_weight_shapes length: {len(combined_shapes)}")
        
        # Perform t-SNE for weights
        tsne_weights = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=42)
        combined_weight_tsne = tsne_weights.fit_transform(combined_weights)        
        
        # UMAP for weights
        umap_weights = UMAP(n_components=2, n_neighbors=30, min_dist=0.1, random_state=42)
        combined_weight_umap = umap_weights.fit_transform(combined_weights)
        
        markers = ['o', 'v']    # Circle for supervised, triangle for unsupervised
        colors = ['blue', 'red']        # Blue for label 0, red for label 1   
        c_colors = plt.cm.jet(np.linspace(0, 1, n_sup + n_unsup)) 
        
        ####################################################################### Plotting TSNE
        plt.figure(figsize=(16, 8))
        
        # Plot concepts
        plt.subplot(1, 2, 1)
        for i in range(0, len(concept_list), n_sup + n_unsup):
            group = concept_list[i:i + n_sup + n_unsup]
            for j in range(len(group)):
                marker = markers[0] if j < n_sup else markers[1]
                color = c_colors[j]
                x_value = i // (n_sup + n_unsup)
                plt.scatter(x_value, group[j], marker=marker, color=color)
        
        
        # Create legend for concepts
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label='Supervised', markerfacecolor='black', markersize=10),
                           plt.Line2D([0], [0], marker='v', color='w', label='Unsupervised', markerfacecolor='black', markersize=10)]
        plt.legend(handles=legend_elements, title='Concept Type')
        plt.title('Concept List Visualization')
        plt.xlabel('Different Samples')
        plt.ylabel('Concept Value')
        plt.grid(True)
        
        # Plot t-SNE for weights
        plt.subplot(1, 2, 2)      
        
        for i in range((n_sup + n_unsup) *idx):
            # first i * n_sup samples superv, other n_unsup *i unsupervis
            marker = markers[int(i >= n_sup* idx)]                
            color = colors[label_list[i]]
            plt.scatter(combined_weight_tsne[i, 0], combined_weight_tsne[i, 1], marker=marker, color=color, label=f'Pred class {0}')
        
        plt.title(f't-SNE of Weights in {phase}')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        
        # Create legend for weights
        legend_elements_weights = [plt.Line2D([0], [0], marker='o', color='w', label='Supervised', markerfacecolor='black', markersize=10),
                                   plt.Line2D([0], [0], marker='v', color='w', label='Unsupervised', markerfacecolor='black', markersize=10)]
        plt.legend(handles=legend_elements_weights, title='Concept Type')
        
        plt.tight_layout()
        
        # Save or show plot
        if savedir:
            if not os.path.exists(savedir):
                os.makedirs(savedir)
            plt.savefig(os.path.join(savedir, f'{phase}_tsne_split.png'))
            print(f'T-SNE saved at {savedir}/{phase}_tsne_split.png')
        
        plt.show()
        
    ############################################### plotting UMAP
        
    # Plotting
    plt.figure(figsize=(8, 8))
    for i in range((n_sup + n_unsup) *idx):
        # first i * n_sup samples superv, other n_unsup *i unsupervis
        marker = markers[int(i >= n_sup* idx)]
        if i < n_sup * idx:
            color_index = i % n_sup
        else:
            color_index = n_sup + (i % n_unsup) 
        color = c_colors[color_index]
        plt.scatter(combined_weight_umap[i, 0], combined_weight_umap[i, 1], marker=marker, color=color, label=f'Pred class {0}')
    
    plt.title(f'UMAP of Weights in {phase}')
    plt.xlabel('UMAP Component 1')
    plt.ylabel('UMAP Component 2')
    
    # Create legend for weights
    legend_elements_weights = [
        plt.Line2D([0], [0], marker='o', color='w', label='Supervised', markerfacecolor='black', markersize=10),
        plt.Line2D([0], [0], marker='v', color='w', label='Unsupervised', markerfacecolor='black', markersize=10)
    ]
    plt.legend(handles=legend_elements_weights, title='Concept Type')
    
    plt.tight_layout()
    
    # Save or show plot
    if savedir:
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        plt.savefig(os.path.join(savedir, f'{phase}_umap_split.png'))
        print(f'UMAP saved at {savedir}/{phase}_umap_split.png')
    
    plt.show()

# %%
