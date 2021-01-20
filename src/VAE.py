import torch
import torch.nn as nn
import torch.nn.functional as F
# import torch.optim as optim
# import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
# import seaborn as sns
from skimage import io
# from matplotlib import pyplot as plt
# from matplotlib import cm
from warnings import warn
import os
import re
import datetime
import gc
import sys
import psutil
# from memory_profiler import profile
# from tqdm import tqdm


def check_mem_usage():
    """
    Check memory used, in GB
    """
    mem_info = psutil.virtual_memory()
    total_mem = mem_info[0]
    avail_mem = mem_info[1]
    mem_used = np.float(total_mem - avail_mem)/np.power(1000, 3)
    now = datetime.datetime.now()
    print(now.strftime("%Y-%m-%d %H:%M:%S") + ' mem used: {} GB'.format(mem_used))

# Classes with callable methods to be used for transforms
class SampleTransform(object):
    """
    Generic Class for doing a transform of image data in a sample dict. Note that this could be extended
    to include transformations on tabular data as well, but for now we're only worried about image data
    """

    def __init__(self, img_transform):
        # img_transform must be subclass of nn.Module
        self.img_transform = img_transform

    def __transform__(self, sample):
        raise NotImplementedError('method __transform__ not defined for this class.')

    def __call__(self, sample):
        image, table_data = sample['image'], sample['table_data']
        transf_image = self.__transform__(image)
        sample_return = {'image': transf_image, 'table_data': table_data}
        return sample_return


class ToTensor(SampleTransform):
    """
    Turns image into a pytorch tensor. See docs for
    torchvision.transforms.ToTensor() for more details
    """

    def __init__(self):
        super().__init__(img_transform=transforms.ToTensor())

    def __transform__(self, image):
        transf_image = self.img_transform.__call__(image)
        return transf_image


class Normalize(SampleTransform):
    """
    Normalize an image to specified mean and std deviation
    """

    def __init__(self, mean, std, inplace=False):
        super().__init__(img_transform=transforms.Normalize(mean, std, inplace))

    def __transform__(self, image):
        transf_image = self.img_transform.__call__(image)
        return transf_image


class RandomHorizontalFlip(SampleTransform):
    """
    Performs random horizontal flip of image with p = 0.5 for the flip
    See docs for torchvision.transforms.RandomHorizontalFlip for more details
    """

    def __init__(self):
        super().__init__(img_transform=transforms.RandomHorizontalFlip())

    def __transform__(self, image):
        transf_image = self.img_transform.__call__(image)
        return transf_image


# basic transform function
# global variables for mean and std dev of intensity based on sandbox notebook
# TODO: setup to calculate these stats based on random sample of images
mean_intensity = np.float(123.5)
std_intensity = np.float(58.2)

basic_transform = transforms.Compose([
        ToTensor(),
        Normalize(mean_intensity, std_intensity),
        RandomHorizontalFlip()
    ])


class XRayDataset(Dataset):
    """
    Xray images dataset. Inspired in part by: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    Used to manage indexing of tabular and image data, as well as do limited transformations of tabular data
    """

    def __init__(self, table_data, root_dir, transform=None):
        """
        Args:
            csv (string): Path to the csv file with tabular data corresponding to images, or pandas dataframe
            root_dir (string): Directory with all the images. Assumes all images in 1 directory, (no subdirectories)
            transform (callable, optional): Optional transform to be applied
                on a sample.
        Notes:
            csv file is read, and then listed images checked against those actually present in root_dir
            only images in the intersection of images listed in csv file and images present in root_dir
            will be used in the XRayDataSet object. Also, transforms could range from initial preprocessing
            for images to be used in training autoencoder to running images through preprocessing and

        """

        if isinstance(table_data, str):
            table_data = pd.read_csv(table_data)
        else:
            try:
                assert isinstance(table_data, pd.DataFrame)
            except:
                raise TypeError('table_data must be valid filepath or pandas DataFrame')

        self.table_data = table_data
        self.root_dir = root_dir

        # subset tabular data only for images in intersection of tabular data and images present in root_dir
        tab_images = self.table_data['Image Index'].to_numpy().astype('str')
        dir_images = os.listdir(root_dir)
        common_images = np.intersect1d(tab_images, dir_images)
        msg = "{0} images common to root_dir ({1} images) and table data ({2} images)"
        print(msg.format(len(common_images), len(dir_images), len(tab_images)))

        if len(common_images) < 1:
            raise ValueError('0 images common between root_dir and table data')

        idx_keep = np.in1d(tab_images, common_images)
        self.table_data = self.table_data.iloc[idx_keep, :]

        self.transform = transform

    def __len__(self):
        return (len(self.table_data))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.table_data.iloc[idx, 0])
        image = io.imread(img_name)
        if len(image.shape) > 2:
            # assuming that first image in image series identical to 2 others in series,
            # and last (4th) is blank as was seen in earlier exploration of 'multi-channel'
            # images. This assumption was based on manual examination of data.
            image = image[:, :, 0]
        # gets 1 row of dataframe if idx of len 1 and returns it as a dict
        table_data = self.table_data.iloc[np.arange(0, 1), :].to_dict(orient='records')[0]['Image Index']

        sample = {'image': image, 'table_data': table_data}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def get_label_names(self, col):
        """
        get unique label names. assume multilabel entries are separated
        by a pipe character
        """
        # get unique label names
        col_contents = self.table_data[col]
        label_names = [re.split('\|', x) for x in col_contents.to_numpy().astype('str')]
        label_names = np.unique(np.array([x for y in label_names for x in y]))
        return label_names

    def one_hot(self, col, label):
        """
        One-hot encode a particular label in categorical data.
        """
        r = re.compile('(\|)*' + label + '(\|)*')
        col_content = self.table_data[col].to_numpy().astype('str')
        vmatch = np.vectorize(lambda x: bool(r.search(x)))
        sel = vmatch(col_content).astype('int32')
        new_col = label
        if new_col in self.table_data.keys():
            warn(new_col + ' is already present in dataframe, replacing')
        self.table_data[new_col] = sel

    def one_hot_all_labels(self, col):
        """
        One hot encode all unique labels in a column. Assume that multilabel entries are
        separated by a pipe | character
        """
        all_labels = self.get_label_names(col)
        for label in all_labels:
            self.one_hot(col, label)

    def get_multi_label(self, col):
        """
        """
        r = re.compile('\|')
        col_content = self.table_data[col].to_numpy().astype('str')
        vmatch = np.vectorize(lambda x: bool(r.search(x)))
        sel = vmatch(col_content).astype('int32')
        new_col = col + '_IsMultiLabel'
        if new_col in self.table_data.keys():
            warn(new_col + ' is already present in dataframe, replacing')
        self.table_data[new_col] = sel

    def train_test_split(self, stratify=None, train_frac=0.8):
        """
        Args:
            stratify = str
        return: a dict of 2 XRayDataset class objects referring to the same image directory
        """

        if isinstance(stratify, str):
            stratify = self.table_data[stratify].to_numpy()

            # get split indices
        n_total = self.__len__()
        all_inds = np.arange(n_total)
        train_inds, test_inds = train_test_split(all_inds, train_size=train_frac, stratify=stratify)
        train_table = self.table_data.iloc[train_inds, :]
        test_table = self.table_data.iloc[test_inds, :]
        n_train = len(train_inds)
        n_test = len(test_inds)
        print("No. Train:{} No. Test:{}".format(n_train, n_test))
        # make and return datasets
        TrainDS = XRayDataset(train_table, self.root_dir, self.transform)
        TestDS = XRayDataset(test_table, self.root_dir, self.transform)
        dict_return = {'train': TrainDS, 'test': TestDS}
        return dict_return


def conv_output_size(W1, S, F, P):
    """
    W1 = input width
    S = Stride
    F = Filter (kernel) size
    P = padding
    returns: W2, or output width
    """
    W2 = np.floor((W1 + 2 * P - F) / S) + 1
    return W2


class Encoder(nn.Module):
    """
    Encoder function. Assumes square image input with 1 channel.
    """

    def __init__(self, c, input_size=1024, latent_dims=20, S=2, F=4, P=1):
        super(Encoder, self).__init__()
        # note: if we want to make this prettier later we can make depth a controllable parameter
        # and loop over depth indices to produce layers or call layers when running the forward method
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=c, kernel_size=F, stride=S, padding=P)
        conv1_out_size = conv_output_size(input_size, S, F, P)
        self.conv2 = nn.Conv2d(in_channels=c, out_channels=c * 2, kernel_size=F, stride=S, padding=P)
        conv2_out_size = conv_output_size(conv1_out_size, S, F, P)
        self.conv3 = nn.Conv2d(in_channels=c * 2, out_channels=c * 4, kernel_size=F, stride=S, padding=P)
        conv3_out_size = conv_output_size(conv2_out_size, S, F, P)
        linear_input_size = (int(c * 4 * conv3_out_size * conv3_out_size))
        self.fc_mu = nn.Linear(in_features=linear_input_size, out_features=(int(latent_dims)))
        self.fc_logvar = nn.Linear(in_features=linear_input_size, out_features=(int(latent_dims)))

    def forward(self, x):
        x = F.softplus(self.conv1(x))
        x = F.softplus(self.conv2(x))
        x = F.softplus(self.conv3(x))
        x = x.view(x.size((int(0))), -1)  # flatten batch of multi-channel feature maps to a batch of feature vectors
        x_mu = self.fc_mu(x)
        x_logvar = self.fc_logvar(x)
        return x_mu, x_logvar


class Decoder(nn.Module):
    """
    Decoder function.
    """
    def __init__(self, c, input_size=1024, latent_dims=20, S=2, F=4, P=1):
        super(Decoder, self).__init__()
        # the below four lines are ugly but quick. In future make this class able to dynamically
        # allocate layers, but first let's get a simple first pass
        conv1_out_size = conv_output_size(input_size, S, F, P)
        conv2_out_size = conv_output_size(conv1_out_size, S, F, P)
        conv3_out_size = conv_output_size(conv2_out_size, S, F, P)
        self.conv3_out_size = int(conv3_out_size)
        self.channels = c
        self.fc1 = nn.Linear(in_features=(int(latent_dims)),
                             out_features=(int(c * 4 * conv3_out_size * conv3_out_size)))
        self.conv3 = nn.ConvTranspose2d(in_channels=c * 4, out_channels=c * 2, kernel_size=F, stride=S, padding=P)
        self.conv2 = nn.ConvTranspose2d(in_channels=c * 2, out_channels=c, kernel_size=F, stride=S, padding=P)
        self.conv1 = nn.ConvTranspose2d(in_channels=c, out_channels=1, kernel_size=F, stride=S, padding=P)

    def forward(self, x):
        c = self.channels
        conv3_out_size = self.conv3_out_size
        x = self.fc1(x)
        x = x.view((x.shape[0], c * 4, conv3_out_size,
                    conv3_out_size))  # unflatten batch of feature vectors to a batch of multi-channel feature maps
        x = F.softplus(self.conv3(x))
        x = F.softplus(self.conv2(x))
        x = self.conv1(x)
        return x


class VariationalAutoencoder(nn.Module):
    def __init__(self, input_size=1024, c=4, latent_dims=20, S=2, F=4, P=1, training=True):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = Encoder(input_size=input_size, c=c, latent_dims=latent_dims, S=S, F=F, P=P)
        self.decoder = Decoder(input_size=input_size, c=c, latent_dims=latent_dims, S=S, F=F, P=P)
        self.training = training

    def set_train_status(self, training):
        self.training = training

    def forward(self, x):
        latent_mu, latent_logvar = self.encoder(x)
        z = self.latent_sample(latent_mu, latent_logvar)
        x_recon = self.decoder(z)
        return x_recon, z, latent_mu, latent_logvar

    def latent_sample(self, mu, logvar):
        if self.training:
            # the reparameterization trick
            # std deviation = sqrt(var) or exp(1/2 logvar)
            std = logvar.mul(0.5).exp_()
            eps = torch.empty_like(std).normal_()
            z = eps.mul(std).add_(mu)
        else:
            # When not training, return the 'expected' position
            z = mu

        return z


def vae_loss(recon_x, x, z, mu, logvar):
    """
    recon_x = reconstructed images of x
    x = original set of images
    z = value of z sampled VariationalAutoEncoder.latent_sample()
    mu = mean in latent space
    logvar = log-variance in latent space
    """
    std_pixels = torch.std(x)
    # calculate log p(x|z) for all pixels. calculated over all pixels over all images
    # normal_dist = torch.distributions.normal.Normal(0, 1)
    # std_norm_xz = (recon_x - x)/std_pixels
    normal_xz = torch.distributions.normal.Normal(recon_x, std_pixels)
    log_pxz = torch.sum(normal_xz.log_prob(x))
    # calculate log p(z). calculated over all images over all latent dims
    normal_z = torch.distributions.normal.Normal(0., 1.)
    log_pz = torch.sum(normal_z.log_prob(z))
    # calculate log q(z). calculated over all images over all latent dims
    # std_norm_zx = (z-mu)/(logvar.exp())
    normal_zx = torch.distributions.normal.Normal(mu, torch.exp(logvar))
    log_qz = torch.sum(normal_zx.log_prob(z))

    total_samples = np.float(recon_x.shape[0])
    # using this sum is same as performping log p(x|z) + log p(z) - log q(z)
    # for all images, and then summing over all images.
    summed_elbo = log_pxz + log_pz - log_qz
    # ELBO = Eq[log p(x|z) + log p(z) - log q(z)]
    elbo = summed_elbo / total_samples
    # we want to maximize elbo so we set the -elbo as the loss
    loss = -elbo
    return loss, elbo, log_pxz

class Trainer(object):

    def __init__(self, XRayDS, stratify=None, train_frac=0.8, learning_rate=1e-6, weight_decay=1e-7):
        """
        XRayDS = XRayDataSet
        stratify = vector of classes to stratify by.
        train_frac = fraction of samples to use for training
        """
        # below lines commented out as I can't figure out whey isinstance(XRayDS, XRayDataset) fails
        #         try:
        #             assert isinstance(XRayDS, XRayDataset)
        #         except:
        #             raise TypeError('XRayDS must be instance of XRayDataset')
        self.SplitData = XRayDS.train_test_split(stratify, train_frac)
        self.Model = VariationalAutoencoder()
        self.optimizer = torch.optim.Adam(self.Model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.running_stats = None
    # @profile
    def train(self, num_epochs=2, batch_size=100):
        """
        Run training for specified number of epochs with specified batch size
        """
        # DataLoaders go through all complete batches
        train_loader = DataLoader(self.SplitData['train'], batch_size=batch_size,
                                  shuffle=True, num_workers=0, drop_last=True)
        test_loader = DataLoader(self.SplitData['test'], batch_size=batch_size,
                                 shuffle=True, num_workers=0, drop_last=True)
        loaders = {'train': train_loader, 'test': test_loader}
        running_stats = {'train': {'avg_elbo': np.array([]), 'avg_log_pxz': np.array([])},
                         'test': {'avg_elbo': np.array([]), 'avg_log_pxz': np.array([])}}
        for epoch in range(num_epochs + 1):
            print('===========================================')
            avg_loss = {'train': None, 'test': None}
            now = datetime.datetime.now()
            print(now.strftime("%Y-%m-%d %H:%M:%S"))
            if epoch == num_epochs:
                print('Final Results')
                do_backprop = False
            else:
                print('Epoch {} / {}'.format(epoch + 1, num_epochs))
                do_backprop = True

            for stage in ['train', 'test']:
                print('###########################################')
                print('__' + stage + '__')
                now = datetime.datetime.now()
                print('Stage Beginning')
                print(now.strftime("%Y-%m-%d %H:%M:%S"))
                if stage == 'train':
                    self.Model.set_train_status(True)
                else:
                    self.Model.set_train_status(False)

                DS = self.SplitData[stage]
                total_elbo = 0.
                total_log_pxz = 0.
                m = 0
                for sample_batched in loaders[stage]:
#                     print('### Batch {} ###'.format(m + 1))
                    x = sample_batched['image']
#                     print('Mem usage pre-forward pass')
#                     check_mem_usage()
                    recon_x, z, mu, logvar = self.Model.forward(x)
#                     print('Mem usage post-forward pass')
#                     check_mem_usage()
                    loss, elbo, log_pxz = vae_loss(recon_x, x, z, mu, logvar)
#                     print('Mem Usage post-loss calculation')
#                     check_mem_usage()
                    # print('ELBO class{}'.format(type(elbo)))
                    # print('log_pxz class{}'.format(type(log_pxz)))
                    total_elbo += elbo.item()
                    total_log_pxz += log_pxz.item()
                    elbo.detach()
                    log_pxz.detach()
#                     print('ELBO: {:.2e} log p(x|z): {:.2e} '.format(elbo.item(), log_pxz.item()))
                    del elbo
                    del log_pxz
                    # print('total_elbo class {}'.format(type(total_elbo)))
                    # print('total_elbo size {}'.format(sys.getsizeof(total_elbo)))
                    if do_backprop:
                        if stage == 'train':
                            self.optimizer.zero_grad()
#                             print('Mem Usage post zero grad')
#                             check_mem_usage()
                            loss.backward()
#                             print('Mem Usage post backprop')
#                             check_mem_usage()
                            self.optimizer.step()
#                             print('Mem Usage post optimizer step')
#                             check_mem_usage()

                    m += 1
                    gc.collect()
                # average ELBO across batches + log p(x|z)
                # since log p(x|z) is summed across all pixels across all samples,
                # the sum across all batches is the joint probability across all batches.
                # dividing total_log_pxz by the number of batches gets a log geometric mean
                # for p(x|z) across batches
                avg_elbo = total_elbo / np.float(m)
                avg_log_pxz = total_log_pxz / np.float(m)

                now = datetime.datetime.now()
                print('### Stage Finished ###')
                print(now.strftime("%Y-%m-%d %H:%M:%S"))
                print('ELBO (avg): {:.2e} log p(x|z): (avg) {:.2e} '.format(avg_elbo, avg_log_pxz))
                running_stats[stage]['avg_elbo'] = np.append(running_stats[stage]['avg_elbo'],
                                                             avg_elbo)
                running_stats[stage]['avg_log_pxz'] = np.append(running_stats[stage]['avg_log_pxz'],
                                                                avg_log_pxz)

        self.running_stats = running_stats
        print('===========================================')
        print('Finished!')
        now = datetime.datetime.now()
        print(now.strftime("%Y-%m-%d %H:%M:%S"))
