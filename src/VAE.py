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

    def __transform__(self, image):
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


class Resize(SampleTransform):
    """
    Resizes image to specified shape. Uses bilinear interpolation.
    """

    def __init__(self, size):
        """
        size = output size of image
        """
        super().__init__(img_transform=transforms.Resize(size=size))

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


# class RandomHorizontalFlip(SampleTransform):
#     """
#     Performs random horizontal flip of image with p = 0.5 for the flip
#     See docs for torchvision.transforms.RandomHorizontalFlip for more details
#     """
#
#     def __init__(self):
#         super().__init__(img_transform=transforms.RandomHorizontalFlip())
#
#     def __transform__(self, image):
#         transf_image = self.img_transform.__call__(image)
#         return transf_image


# # basic transform function
# # global variables for mean and std dev of intensity based on sandbox notebook
# # TODO: setup to calculate these stats based on random sample of images
# mean_intensity = np.float32(123.5)
# std_intensity = np.float32(58.2)
# note here we adjust for a mean and std after calling ToTensor. 
mean_intensity = np.float32(0.4746)
std_intensity = np.float32(0.2709)

def create_transform(resize_width=512, mean_intensity=mean_intensity, std_intensity=std_intensity):
    basic_transform = transforms.Compose([
        ToTensor(),
        Resize(size=(resize_width, resize_width)),
        Normalize(mean_intensity, std_intensity)
    ])
    return basic_transform

def create_transform_zero_one_norm(resize_width=512):
    basic_transform = transforms.Compose([
        ToTensor(),
        Resize(size=(resize_width, resize_width))
    ])
    return basic_transform


# basic_transform = transforms.Compose([
#         ToTensor(),
#         Normalize(mean_intensity, std_intensity)
#     ])


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
        label_names = [re.split(r'\|', x) for x in col_contents.to_numpy().astype('str')]
        label_names = np.unique(np.array([x for y in label_names for x in y]))
        return label_names

    def one_hot(self, col, label):
        """
        One-hot encode a particular label in categorical data.
        """
        r = re.compile(r'(?<!\w)\|*' + re.escape(label) + r'\|*(?!\w)')
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
        r = re.compile(r'\|')
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

    def __init__(self, c, input_size=64, latent_dims=16, S=2, F=4, P=1, n_conv = 2, c_mul = 2, use_batch_norm=True):
        """
        c = number of output channels after first convolution. This number is multiplied by c_mul in each layer
        input_size = width of square image
        latent_dims = number of latent dimensions in autoencoder. produced by fully connected layers producing mu and logvar
        S = stride
        F = filter size, for square filter
        P = padding.
        n_conv = number of convolution layers
        c_mul = factor to multiply number of channels by for each convolution after first
        """
        super(Encoder, self).__init__()
        self.conv_layers = nn.ModuleList()
        self.use_batch_norm = use_batch_norm
        if self.use_batch_norm:
            self.norm_layers = nn.ModuleList()

        for i in range(n_conv):
            if i == 0:
                in_channels_i = 1
                out_channels_i = c
                conv_out_size = conv_output_size(input_size, S, F, P)
            else:
                conv_out_size = conv_output_size(conv_out_size, S, F, P)
                out_channels_i = out_channels_i*c_mul
            # Conv2D layer
            new_layer = nn.Conv2d(in_channels=in_channels_i,
                                  out_channels=out_channels_i,
                                  kernel_size=F,
                                  stride=S,
                                  padding=P)
            self.conv_layers.append(new_layer)
            if self.use_batch_norm:
                # 2d BatchNorm layer
                bn_layer = nn.BatchNorm2d(num_features=out_channels_i)
                self.norm_layers.append(bn_layer)

            in_channels_i = out_channels_i

        if self.use_batch_norm:
            assert len(self.norm_layers) == len(self.conv_layers)
        
        linear_input_size = int(out_channels_i*conv_out_size*conv_out_size)
        # self.fc0 = nn.Linear(in_features=linear_input_size, out_features=(int(fc0_dims)))
        # if self.use_batch_norm:
        #     self.fc0_bn = nn.BatchNorm1d(num_features=int(fc0_dims))
        # self.fc_mu = nn.Linear(in_features=fc0_dims, out_features=(int(latent_dims)))
        # self.fc_logvar = nn.Linear(in_features=fc0_dims, out_features=(int(latent_dims)))
        self.fc_mu = nn.Linear(in_features=linear_input_size, out_features=(int(latent_dims)))
        self.fc_logvar = nn.Linear(in_features=linear_input_size, out_features=(int(latent_dims)))

    def forward(self, x):
        # expect a sequence of convolutional layers, followed by 1 fc layer
        # followed by two separate FC layers.
        # output of final conv layer is fed to distinct FC layers
        for i in range(len(self.conv_layers)):
            conv_i = self.conv_layers[i]
            x = conv_i(x)
            if self.use_batch_norm:
                bn_i = self.norm_layers[i]
                x = bn_i(x)
            x = F.relu(x)

        x = x.view(x.size((int(0))), -1)
        # x = self.fc0(x)
        # if self.use_batch_norm:
        #     x = self.fc0_bn(x)
        # x = F.relu(x)
        # x = self.fc0(x)
        x_mu = self.fc_mu(x)
        x_logvar = self.fc_logvar(x)
        
        return x_mu, x_logvar
    

def transposed_conv_out_size(W, S, F, P):
    """
    W = input size
    S = stride
    F = filter size
    P = padding
    returns: output width for transposed convolution
    
    Note: This only works if no output padding or dilation added in pytorch
    """
    O = (W - 1)*S + F - 2*P
    
    return O


class Decoder(nn.Module):
    """
    Decoder function. 
    
    Notes:
    
    Note that in terms of convolutions, we attempt to do a mirror image of the encoder function
    given identical arguments for c, input_size = output_size, latent_dims, S, F, P, n_conv, and c_mul.
    If these arguments are not identical to those in encoder function, encoder output will not be correctly
    handled by decoder. Returns square image of size (input_size, input_size) if (W + 2P - K)/S is an integer
    for each convolution operation.
    
    input_size = size of input to encoder function
    latent_dims = number of latent dims
    S = stride
    F = filter size
    P = padding
    n_conv = number of transposed convolutional layers
    c_mul = factor to divide # channels by after each transposed convolution
    output = 'Gaussian' for gaussian output, 'Sigmoid' for output between 0 and 1. The former case just has a final
    linear transposed convolution to get to the desired output size, while sigmoid puts a sigmoid on top of it
    
    returns: square image of size (input_size, input_size)
    """
    def __init__(self, c, input_size=64, latent_dims=16, S=2, F=4, P=1, n_conv=2, c_mul=2, output='Gaussian', use_batch_norm=True):
        super(Decoder, self).__init__()
        tconv_layer_list = []
        self.tconv_layers = nn.ModuleList()
        self.use_batch_norm = use_batch_norm
        if self.use_batch_norm:
            bnorm_layer_list = []
            self.bnorm_layers = nn.ModuleList()
             
        for i in range(n_conv):
            # first layer defined is the last transposed convolution layer
            # we loop until the first transposed convolution layer, which is to mirror the
            # last convolution layer prior to the linear layer
            if i == 0:
                out_channels_i = 1
                in_channels_i = c
                conv_out_size = conv_output_size(input_size, S, F, P)
            else:
                conv_out_size = conv_output_size(conv_out_size, S, F, P)
                out_channels_i = in_channels_i
                in_channels_i = in_channels_i*c_mul

            if i != 0:
                # batchnorm layers are appended to list prior to tconv layers
                # after first iteration of loop. Reversing order of layers
                # in the ModuleList, we get tconv followed by batch norm for
                # all tconv operations except last
                if self.use_batch_norm:
                    tconv_bn_layer = nn.BatchNorm2d(num_features=out_channels_i)
                    bnorm_layer_list.append(tconv_bn_layer)

            tconv_layer = nn.ConvTranspose2d(in_channels=in_channels_i,
                                             out_channels=out_channels_i,
                                             kernel_size=F,
                                             stride=S,
                                             padding=P)
            tconv_layer_list.append(tconv_layer)

        for i in range(len(tconv_layer_list)):
            # we now append the layers defined in tconv_layer_list in reverse order
            self.tconv_layers.append(tconv_layer_list.pop())
        
        if self.use_batch_norm:
            for i in range(len(bnorm_layer_list)):
                self.bnorm_layers.append(bnorm_layer_list.pop())

            assert len(self.bnorm_layers) == (len(self.tconv_layers) - 1)
            
        # check that output size is OK
        try:
            for i in range(n_conv):
                if i == 0:
                    transp_conv_in_size = conv_out_size
                else:
                    transp_conv_in_size = transp_conv_out_size

                transp_conv_out_size = transposed_conv_out_size(transp_conv_in_size, S, F, P)
                
            if not transp_conv_out_size == input_size:
                assert False
                
        except:
            msg = 'final transposed convolution output size {} does not match input size {}'
            raise ValueError(msg.format(transp_conv_out_size, input_size))

        # # last but not least add fully connected layers.
        # self.fc1 = nn.Linear(in_features=int(latent_dims),
        #                      out_features=int(fc0_dims))
        fc1_out_size=int(in_channels_i*conv_out_size**2)
        self.fc1 = nn.Linear(in_features=int(latent_dims),
                            out_features=fc1_out_size)
        # fc0_out_size=int(in_channels_i*conv_out_size**2)
        # self.fc0 = nn.Linear(in_features=int(fc0_dims),
        #                      out_features=fc0_out_size)
        if self.use_batch_norm:
            # define batch normalization layers for the FC layers
            self.fc1_bn = nn.BatchNorm1d(num_features=fc1_out_size)
            # self.fc1_bn = nn.BatchNorm1d(num_features=fc0_dims)
        #     self.fc0_bn = nn.BatchNorm1d(num_features=fc0_out_size)
        # define output type as binary or gaussian
        self.output = output
        # keep useful numbers
        self.conv_out_size = conv_out_size
        self.in_channels = in_channels_i
                             
    def forward(self, x):
        # # run linear layer
        x = self.fc1(x)
        if self.use_batch_norm:
            x = self.fc1_bn(x)
        x = F.relu(x)
        # x = self.fc0(x)
        # if self.use_batch_norm:
        #     x = self.fc0_bn(x)
        # x = F.relu(x)
        # x = self.fc1(x)
        # x = self.fc0(x)
        # reshape linear layer output for conv layers
        conv_out_size = int(self.conv_out_size)
        in_channels = int(self.in_channels)
        x = x.view((x.shape[0], in_channels, conv_out_size, conv_out_size))
        # run through conv layers
        for i in range(len(self.tconv_layers)):
            conv_i = self.tconv_layers[i]
            x = conv_i(x)
            # only do batch normalization + relu up to penultimate layer output
            if i < (len(self.tconv_layers) - 1):
                if self.use_batch_norm:
                    bnorm_i = self.bnorm_layers[i]
                    x = bnorm_i(x)
                x = F.relu(x)
            # is_tconv = isinstance(f, torch.nn.modules.ConvTranspose2d)
            # last_layer = i == len(self.tconv_layers) - 1
            
        # use sigmoid if binary output desired
        if self.output == 'Gaussian':
            x = x
        elif self.output == 'Sigmoid':
            x = torch.sigmoid(x)
        else:
            raise ValueError('output should be specified as Gaussian or Sigmoid')
            
        return x


class VariationalAutoEncoder(nn.Module):
    def __init__(self, input_size=64, c=4, latent_dims=16, S=2, F=4, P=1, c_mul=2, n_conv=2, training=True, output='Gaussian', use_batch_norm=True):
        super(VariationalAutoEncoder, self).__init__()
        # self.encoder = Encoder(input_size=input_size, c=c, fc0_dims=fc0_dims, latent_dims=latent_dims, S=S, F=F, P=P, c_mul=c_mul, n_conv=n_conv, use_batch_norm=use_batch_norm)
        # self.decoder = Decoder(input_size=input_size, c=c, fc0_dims=fc0_dims, latent_dims=latent_dims, S=S, F=F, P=P, c_mul=c_mul, n_conv=n_conv, output = output, use_batch_norm=use_batch_norm)
        self.encoder = Encoder(input_size=input_size, c=c, latent_dims=latent_dims, S=S, F=F, P=P, c_mul=c_mul, n_conv=n_conv, use_batch_norm=use_batch_norm)
        self.decoder = Decoder(input_size=input_size, c=c, latent_dims=latent_dims, S=S, F=F, P=P, c_mul=c_mul, n_conv=n_conv, output = output, use_batch_norm=use_batch_norm)
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


def vae_loss(recon_x, x, mu, logvar):
    """
    Calculates ELBO with Gaussian likelihood for p(x|z)
    recon_x = reconstructed images of x
    x = original set of images
    z = value of z sampled VariationalAutoEncoder.latent_sample()
    mu = mean in latent space
    logvar = log-variance in latent space
    """
    # std_pixels = torch.std(x)
    # calculate log p(x|z) for all pixels. calculated over all pixels over all images
    # normal_dist = torch.distributions.normal.Normal(0, 1)
    # std_norm_xz = (recon_x - x)/std_pixels
    # we set the standard deviation to 1 to remove stochasticity from reconstruction error.
    # it may well be the case that variation of pixels allows for very tolerant log probability
    # normal_xz = torch.distributions.normal.Normal(recon_x, 1)
    # normal_xz = torch.distributions.normal.Normal(recon_x, 1)
    # log_pxz = torch.sum(normal_xz.log_prob(x))
    # we approximate elbo with MSE loss as log p(X|Z). Normal distribution should effectively be the same
    # but the only working examples of VAEs I've seen for X ray data use MSELoss
    mse_loss = nn.MSELoss(reduction='sum')(recon_x, x)
    # calculate KL divergence between q(z) and p(z).
    kldivergence = -0.5*torch.sum(1. + logvar - mu.pow(2) - logvar.exp())
    # calculate elbo
    #elbo = log_pxz - kldivergence
    # calculate loss
    loss = mse_loss + kldivergence
    # elbo_avg = torch.mean(elbo)
    # loss = -elbo_avg
    elbo = -loss
    # avg_log_pxz = torch.mean(log_pxz)
    # return loss, elbo_avg, avg_log_pxz
    log_pxz = -mse_loss
    return loss, elbo, log_pxz

# def vae_loss_CE(recon_x, x, mu, logvar):
#     """
#     Calculates ELBO with bernoulli log likelihood for p(x|z)
#     recon_x = reconstructed images of x
#     x = original set of images
#     z = value of z sampled VariationalAutoEncoder.latent_sample()
#     mu = mean in latent space
#     logvar = log-variance in latent space
#     """
#     # std_pixels = torch.std(x)
#     # calculate log p(x|z) for all pixels. calculated over all pixels over all images
#     # normal_dist = torch.distributions.normal.Normal(0, 1)
#     # std_norm_xz = (recon_x - x)/std_pixels
#     bernoulli_dist = torch.distributions.bernoulli.Bernoulli(x)
#     log_pxz = torch.sum(bernoulli_dist.log_prob(recon_x), (1, 2, 3))

#     # KL Divergence for gaussian mu and logvar
#     kldivergence = -0.5*torch.sum(1. + logvar - mu.pow(2) - logvar.exp())
#     # we want to maximize elbo so we set the -elbo as the loss
#     elbo = log_pxz - kldivergence
#     # elbo_avg = torch.mean(elbo)
#     # loss = -elbo_avg
#     loss = -elbo
#     # avg_log_pxz = torch.mean(log_pxz)
#     # return loss, elbo_avg, avg_log_pxz
#     return loss, elbo, log_pxz


class Trainer(object):

    def __init__(self, 
                 XRayDS, 
                 Model=VariationalAutoEncoder(),
                 stratify=None, 
                 train_frac=0.8, 
                 learning_rate=1e-3, 
                 weight_decay=1e-5, 
                 use_GPU = True):
        """
        XRayDS = XRayDataSet
        Model = VariationalAutoEncoder object
        stratify = vector of classes to stratify by.
        train_frac = fraction of samples to use for training
        learning_rate = learning rate
        weight_decay = weight decay
        use_GPU = whether to use GPU

        Description: Handles training of VariationalAutoEncoder object given an XRayDS object
        Notes: ADAM optimizer used
        """
        # below lines commented out as I can't figure out whey isinstance(XRayDS, XRayDataset) fails
        #         try:
        #             assert isinstance(XRayDS, XRayDataset)
        #         except:
        #             raise TypeError('XRayDS must be instance of XRayDataset')
        self.SplitData = XRayDS.train_test_split(stratify, train_frac)
        self.Model = Model
        
        if use_GPU:
            if not torch.cuda.is_available():
                warn('torch.cuda.is_available() returned False, using CPU')
            else:
                print('using GPU')
                
            device = torch.device("cuda:0" if use_GPU and torch.cuda.is_available() else "cpu")
            self.Model = self.Model.to(device)
            
        else:
            device = 'cpu'
            print('using CPU')
            
        self.optimizer = torch.optim.Adam(self.Model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.running_stats = None
        self.device = device
        
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
            print('###########################################')
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
                print('===========================================')
                print('__' + stage + '__')
                now = datetime.datetime.now()
                print('Stage Beginning')
                print(now.strftime("%Y-%m-%d %H:%M:%S"))
                if stage == 'train':
                    self.Model.set_train_status(True)
                else:
                    self.Model.set_train_status(False)

                DS = self.SplitData[stage]
                # total elbo and total_log_pxz are used to calculate
                # averaged elbo and log_pxz across multiple batches
                total_elbo = torch.tensor([0.])
                total_log_pxz = torch.tensor([0.])
                m = 0
                for sample_batched in loaders[stage]:
                    x = sample_batched['image']
                    x = x.to(self.device)
                    recon_x, z, mu, logvar = self.Model.forward(x)
                    # elbo and log_pxz are really averages across batch
                    loss, elbo, log_pxz = vae_loss(recon_x, x, mu, logvar)
                    total_elbo += elbo.item()
                    total_log_pxz += log_pxz.item()
                    elbo.detach()
                    log_pxz.detach()
                    del elbo
                    del log_pxz
        
                    if do_backprop:
                        if stage == 'train':
                            self.optimizer.zero_grad()
                            loss.backward()
                            self.optimizer.step()

                    m += 1
                    gc.collect()
                # average ELBO across batches + log p(x|z)
                # since log p(x|z) is summed across all pixels across all samples,
                # the sum across all batches is the joint probability across all batches.
                # dividing total_log_pxz by the number of batches gets a log geometric mean
                # for p(x|z) across batches
                avg_elbo = (total_elbo / torch.tensor(m)).numpy()[0]
                avg_log_pxz = (total_log_pxz / torch.tensor(m)).numpy()[0]

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
