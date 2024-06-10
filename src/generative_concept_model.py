#%%
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
# import torchvision.models as models
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
from tqdm import tqdm
import time
import argparse
import json
import yaml
import pandas as pd

from src.utils import util_path
from src.utils import util_nn
from src.utils import util_data
from src.models import models

column_width_pt = 516.0
pt_to_inch = 1 / 72.27
column_width_inches = column_width_pt * pt_to_inch
aspect_ratio = 4 / 3
sns.set(style="whitegrid", font_scale=1.6, rc={"figure.figsize": (column_width_inches, column_width_inches / aspect_ratio)})
#%% 

def parse_args():
    parser = argparse.ArgumentParser(description=".")
    parser.add_argument("-cfg_file", required=True, type=str, help="Path to the JSON file.")
    return parser.parse_args()

if __name__ == "__main__":
#%%
    # args = parse_args()

    cfg_file = '/home/riccardo/Github/generative-cem/configs/sequential_10100.yaml'
    # Load JSON file. from the disk.
    with open(cfg_file) as file:        # args.cfg_file
        cfg = yaml.load(file, Loader=yaml.FullLoader)

    exp_name = cfg['exp_name']
    seed = cfg['seed']
    device_name = cfg['device']

    # Dataset.
    cfg_data = cfg['DATA']
    dataset_name = cfg_data['dataset_name']
    data_dir = cfg_data['data_dir']
    file_name = cfg_data['file_name']
    file_type = cfg_data['file_type']
    img_size = cfg_data['img_size']
    n_channels = cfg_data['n_channels']
    n_classes = cfg_data['n_classes']
    n_concepts = cfg_data['n_concepts']
    n_embed_concepts = cfg_data['n_embed_concepts']
    n_model_concepts = cfg_data['n_model_concepts']

    # Directories.
    cfg_dir = cfg['REPORTS']
    models_dir = os.path.join(cfg_dir['models_dir'], dataset_name, exp_name)
    logs_dir = os.path.join(cfg_dir['logs_dir'], dataset_name, exp_name)
    reports_dir = os.path.join(cfg_dir['reports_dir'], dataset_name, exp_name)
    util_path.create_dir(models_dir)
    util_path.create_dir(logs_dir)
    util_path.create_dir(reports_dir)

    # Save the config file in logs.
    with open(os.path.join(logs_dir, 'config.yaml'), 'w') as file:
        yaml.dump(cfg, file)

    # Model.
    cfg_model = cfg['MODEL']
    freeze_encoder = cfg_model['freeze_encoder']
    pretrained_model_path = cfg_model['pretrained_model_path']

    # Parameters.
    cfg_train = cfg['TRAINING']
    num_epochs = cfg_train['num_epochs']
    batch_size = cfg_train['batch_size']
    learning_rate = cfg_train['learning_rate']
    warmup_epoch = cfg_train['warmup_epochs']
    early_stopping = cfg_train['early_stopping']
    # Loss weights.
    weight_concept_loss = cfg_train['weight_concept_loss']
    weight_task_loss = cfg_train['weight_task_loss']
    weight_rec_loss = cfg_train['weight_rec_loss']
    weight_lat_loss = cfg_train['weight_lat_loss']
    weight_orth_loss = cfg_train['weight_orth_loss']

     # Dataset.
    if 'dsprites' in dataset_name:
        datasets = {
            step: util_data.dSpritesDataset(data_dir=data_dir, fname=f'{file_name}_{step}.{file_type}', use_concepts=True)
            for step in ['train', 'val', 'test']
        }
    else:
        raise ValueError(f"Dataset {dataset_name} not implemented.")

    # Creating DataLoaders.
    data_loaders = {
        step: DataLoader(datasets[step], batch_size=batch_size, shuffle=True)
        for step in ['train', 'val', 'test']
    }

    # Sample one batch.
    sample = next(iter(data_loaders['train']))

    assert n_channels == sample['image'].shape[1], "The number of channels in the dataset is different from the one specified in the arguments."
    assert img_size == sample['image'].shape[2], "The image size in the dataset is different from the one specified in the arguments."
    assert n_concepts ==  sample['concepts'].shape[1], "The number of concepts in the dataset is different from the one specified in the arguments."

    from src.models.models import select_model, Encoder, MLP, IndependentMLP, Decoder, End2End
    # model = models.select_model....
    model = select_model(exp_name=exp_name, input_size=(n_channels, img_size, img_size), num_concepts=n_concepts, \
                                num_embed_for_concept=n_embed_concepts, num_model_concepts=n_model_concepts, num_classes=n_classes)

    # if pretrained_model_path is not None:
    #     model.load_state_dict(torch.load(pretrained_model_path))

    # Train - Paul Kalkbrenner
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    model = model.to(device)
#%%
    model, train_history, mse_errors_dict = util_nn.train_model(
        model=model,
        data_loaders=data_loaders,
        learning_rate=learning_rate,
        weights = {'concept': weight_concept_loss,'task': weight_task_loss, \
                   'rec': weight_rec_loss, 'lat': weight_lat_loss, 'orth': weight_orth_loss},
        num_epochs=num_epochs,
        early_stopping=early_stopping,
        warmup_epoch=warmup_epoch,
        models_dir=models_dir,
        device=device,
        freeze_encoder=freeze_encoder,
    )
#%%
    # Save history.
    train_history.to_csv(os.path.join(reports_dir, 'train_history.csv'), index=False)
    
    # Evaluate.
    print("Evaluating the model.")
    test_history = util_nn.evaluate(model=model, data_loader=data_loaders['test'], device=device)
    test_history.to_csv(os.path.join(reports_dir, 'test_history.csv'), index=False)

    # Extract embeddings.
    print("Extracting embeddings.")
    c_train = util_nn.predict(model, data_loaders['train'], device)
    c_val = util_nn.predict(model, data_loaders['val'], device)
    c_test = util_nn.predict(model, data_loaders['test'], device)

    # Save c_train, c_val, c_test dividing the splits.
    outdir = os.path.join('home/riccardo/Github/generative-cem/data/', exp_name)
    util_path.create_dir(outdir)
    np.save(os.path.join(outdir, 'c_train.npy'), c_train)
    np.save(os.path.join(outdir, 'c_val.npy'), c_val)
    np.save(os.path.join(outdir, 'c_test.npy'), c_test)

    # Plot.
    util_nn.plot_training(train_history, reports_dir, loss_names=['train_concept_loss', 'train_task_loss', 'train_rec_loss', 'train_lat_loss', 'train_total_loss'])
    util_nn.plot_training(train_history, reports_dir, loss_names=['val_concept_loss', 'val_task_loss', 'val_rec_loss', 'val_total_loss'])
    util_nn.plot_reconstructions(reports_dir, model, data_loaders['test'], device, num_images=5)

    mse_data_loaders = {
    step: DataLoader(datasets[step], batch_size=1, shuffle=True)
    for step in ['train', 'val', 'test']
    }
    # Correlations and mse
    util_nn.calculate_and_save_distances(model, mse_data_loaders, device, reports_dir)
    util_nn.plot_mse_values(reports_dir +'/distances.pkl')
    util_nn.calculate_and_save_correlations(model, mse_data_loaders, device, reports_dir)
    util_nn.plot_correlations(reports_dir +'/correlations.pkl')
    
    
    # # Save MSE erros history to CSV
    # with open(os.path.join(reports_dir, 'mse_errors.csv'), 'w') as f:
    #     for epoch in mse_errors_dict:
    #         for phase in mse_errors_dict[epoch]:
    #             f.write(f'EPOCH {epoch + 1} - Phase: {phase}\n')
    #             df = pd.DataFrame(mse_errors_dict[epoch][phase])
    #             df.to_csv(f, index=False)
    #             f.write('\n')
    
    # Plot MSE for each supervised concepts  vs epochs for visualizations 
    util_nn.plot_sup_unsup_mse(mse_errors_dict, preds_concepts_size=n_concepts, unsup_concepts_size=n_model_concepts, 
                           save_dir=reports_dir)
    
    print("May the force be with you!")
# %%
