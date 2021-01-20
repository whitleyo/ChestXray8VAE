import VAE
import os
import torch
import numpy as np
from warnings import warn
import pandas as pd

# set random seed
np.random.seed(42)

# setup directories
# top_dir = 'D:/projects/chest_XRay_8'
top_dir = '/home/owhitley/projects/chest_xray_8'
data_dir = os.path.join(top_dir, 'data')
image_dir = os.path.join(data_dir, 'images')
output_dir = os.path.join(top_dir, 'results')
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

# setup dataset
table_data = pd.read_csv(os.path.join(data_dir, 'Data_Entry_2017_v2020.csv'))
table_data = table_data.iloc[0:1000, :]
VAE_DS = VAE.XRayDataset(table_data=table_data,
                         root_dir=image_dir,
                         transform=VAE.basic_transform)

# setup trainer and train
VAE_Trainer = VAE.Trainer(XRayDS=VAE_DS)
n_epochs = 5
VAE_Trainer.train(num_epochs=n_epochs, batch_size=100)
# save model state
FinalModel = VAE_Trainer.Model
torch.save(FinalModel.state_dict(), os.path.join(output_dir, 'VAE_state_dict_trained.pt'))

