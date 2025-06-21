import VAE
import os
import torch
import numpy as np
from warnings import warn
import pandas as pd

# set random seed
np.random.seed(42)

# setup directories
# top_dir = 'D:\\projects\\chest_XRay_8'
top_dir = '/home/whitleyo/projects/ChestXray8VAE'
data_dir = os.path.join(top_dir, 'data')
image_dir = os.path.join(data_dir, 'images')
output_dir = os.path.join(top_dir, 'results')
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

# setup dataset
table_data = pd.read_csv(os.path.join(data_dir, 'Data_Entry_2017_v2020.csv'))
table_data = table_data.iloc[0:5000, :] # take first 5k. Should be sufficient for decent-ish performance
VAE_DS = VAE.XRayDataset(table_data=table_data,
                         root_dir=image_dir,
                         transform=VAE.create_transform_zero_one_norm(resize_width=256))

# setup trainer and train
VAE_Trainer = VAE.Trainer(XRayDS=VAE_DS,
                          Model=VAE.VariationalAutoEncoder(input_size=256,
                                                           # fc0_dims=1024,
                                                           latent_dims=512,
                                                           n_conv=4,
                                                           F=4,
                                                           P=1,
                                                           S=2,
                                                           c=32,
                                                           use_batch_norm=False,
                                                           output='Sigmoid'),
                          learning_rate=1e-3)
print('model')
print(VAE_Trainer.Model)
VAE_Trainer.train(num_epochs=50, batch_size=20)
# VAE_Trainer.train(num_epochs=100)
# save model state
FinalModel = VAE_Trainer.Model
torch.save(VAE_Trainer, os.path.join(output_dir, 'VAE_trainer.pt'))
torch.save(FinalModel.state_dict(), os.path.join(output_dir, 'VAE_state_dict_trained.pt'))
torch.save(VAE_Trainer.running_stats, os.path.join(output_dir, 'VAE_running_stats.pt'))
