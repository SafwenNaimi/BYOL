# -*- coding: utf-8 -*-
"""
Created on Fri May 14 11:13:07 2021

@author: Safwen
"""

import os
import argparse
import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision import models, datasets
import numpy as np
from PIL import ImageFilter, ImageOps
import torchvision.transforms as transforms
from collections import defaultdict
from torchvision import models
#from models.resnet_base_network import ResNet18
# distributed training
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

#from byol_2 import BYOL
from data.transforms import get_simclr_data_transforms
import yaml
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
from torch.utils.data.dataloader import default_collate
from torch.utils.data.dataloader import DataLoader
import copy
import random
from functools import wraps
from simclr_tran import TransformsSimCLR

import torch
from torch import nn
import torch.nn.functional as F

def default(val, def_val):
    return def_val if val is None else val


def flatten(t):
    return t.reshape(t.shape[0], -1)


def singleton(cache_key):
    def inner_fn(fn):
        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            instance = getattr(self, cache_key)
            if instance is not None:
                return instance

            instance = fn(self, *args, **kwargs)
            setattr(self, cache_key, instance)
            return instance

        return wrapper

    return inner_fn


# loss fn
class GaussianBlur(object):
    """Gaussian Blur version 2"""

    def __call__(self, x):
        sigma = np.random.uniform(0.1, 2.0)
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

def loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)


# augmentation utils


class RandomApply(nn.Module):
    def __init__(self, fn, p):
        super().__init__()
        self.fn = fn
        self.p = p

    def forward(self, x):
        if random.random() > self.p:
            return x
        return self.fn(x)


# exponential moving average


class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(
        current_model.parameters(), ma_model.parameters()
    ):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)


# MLP class for projector and predictor


class MLP(nn.Module):
    def __init__(self, dim, projection_size, hidden_size=4096):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, projection_size),
        )

    def forward(self, x):
        return self.net(x)


# a wrapper class for the base neural network
# will manage the interception of the hidden layer output
# and pipe it into the projecter and predictor nets


class NetWrapper(nn.Module):
    def __init__(self, net, projection_size, projection_hidden_size, layer=-2):
        super().__init__()
        self.net = net
        self.layer = layer

        self.projector = None
        self.projection_size = projection_size
        self.projection_hidden_size = projection_hidden_size

        self.hidden = None
        self.hook_registered = False

    def _find_layer(self):
        if type(self.layer) == str:
            modules = dict([*self.net.named_modules()])
            return modules.get(self.layer, None)
        elif type(self.layer) == int:
            children = [*self.net.children()]
            return children[self.layer]
        return None

    def _hook(self, _, __, output):
        self.hidden = flatten(output)

    def _register_hook(self):
        layer = self._find_layer()
        assert layer is not None, f"hidden layer ({self.layer}) not found"
        handle = layer.register_forward_hook(self._hook)
        self.hook_registered = True

    @singleton("projector")
    def _get_projector(self, hidden):
        _, dim = hidden.shape
        projector = MLP(dim, self.projection_size, self.projection_hidden_size)
        return projector.to(hidden)

    def get_representation(self, x):
        if not self.hook_registered:
            self._register_hook()

        if self.layer == -1:
            return self.net(x)

        _ = self.net(x)
        hidden = self.hidden
        self.hidden = None
        assert hidden is not None, f"hidden layer {self.layer} never emitted an output"
        return hidden

    def forward(self, x):
        representation = self.get_representation(x)
        projector = self._get_projector(representation)
        projection = projector(representation)
        return projection


# main class


class BYOL(nn.Module):
    def __init__(
        self,
        net,
        image_size,
        hidden_layer=-2,
        projection_size=256,
        projection_hidden_size=4096,
        augment_fn=None,
        moving_average_decay=0.99,
    ):
        super().__init__()

        self.online_encoder = NetWrapper(
            net, projection_size, projection_hidden_size, layer=hidden_layer
        )
        self.target_encoder = None
        self.target_ema_updater = EMA(moving_average_decay)

        self.online_predictor = MLP(
            projection_size, projection_size, projection_hidden_size
        )

        # send a mock image tensor to instantiate singleton parameters
        self.forward(torch.randn(2, 3, image_size, image_size), torch.randn(2, 3, image_size, image_size))

    @singleton("target_encoder")
    def _get_target_encoder(self):
        target_encoder = copy.deepcopy(self.online_encoder)
        return target_encoder

    def reset_moving_average(self):
        del self.target_encoder
        self.target_encoder = None

    def update_moving_average(self):
        assert (
            self.target_encoder is not None
        ), "target encoder has not been created yet"
        update_moving_average(
            self.target_ema_updater, self.target_encoder, self.online_encoder
        )

    def forward(self, image_one, image_two):
        online_proj_one = self.online_encoder(image_one)
        online_proj_two = self.online_encoder(image_two)

        online_pred_one = self.online_predictor(online_proj_one)
        online_pred_two = self.online_predictor(online_proj_two)

        with torch.no_grad():
            target_encoder = self._get_target_encoder()
            target_proj_one = target_encoder(image_one)
            target_proj_two = target_encoder(image_two)

        loss_one = loss_fn(online_pred_one, target_proj_two.detach())
        loss_two = loss_fn(online_pred_two, target_proj_one.detach())

        loss = loss_one + loss_two
        return loss.mean()


def get_model(params,pretrained=True):
    if params.network=='resnet18':
        model = models.resnet18(pretrained=pretrained)
    
        model.fc = nn.Linear(in_features=model.fc.in_features,out_features=5,bias=True)
        return model


class FlowersDataset(Dataset):
    """ Flowers Dataset Class loader """
    
    def __init__(self,cfg, annotation_file,data_type='train', \
                 transform=None):
        
        """
        Args:
            image_dir (string):  directory with images
            annotation_file (string):  csv/txt file which has the 
                                        dataset labels
            transforms: The trasforms to apply to images
        """
        
        self.data_path = os.path.join(cfg.root_path,cfg.data_path,cfg.imgs_dir)
        self.label_path = os.path.join(cfg.root_path,cfg.data_path,cfg.labels_dir,annotation_file)
        self.transform=transform
        self.pretext = cfg.pretext
        if self.pretext == 'rotation':
            self.num_rot = cfg.num_rot
        self._load_data()

    def _load_data(self):
        '''
        function to load the data in the format of [[img_name_1,label_1],
        [img_name_2,label_2],.....[img_name_n,label_n]]
        '''
        self.labels = pd.read_csv(self.label_path)
        
        self.loaded_data = []
#        self.read_data=[]
        for i in range(self.labels.shape[0]):
            img_name = self.labels['Filename'][i]#os.path.join(self.data_path, self.labels['Category'][i],self.labels['FileName'][i])
            #print(img_name)
            #data.append(io.imread(os.path.join(self.image_dir, self.labels['img_name'][i])))
            label = self.labels['Label'][i]
            img = Image.open(img_name)
            self.loaded_data.append((img,label,img_name))
            img.load()#This closes the image object or else you will get too many open file error
#            self.read_data.append((img,label))

    def __len__(self):
        return len(self.loaded_data)

    def __getitem__(self, idx):

        idx = idx % len(self.loaded_data)
        img,label,img_name = self.loaded_data[idx]
#        img = io.imread(img_name)
#        img = Image.open(img_name)   
        img,label = self._read_data(img,label)
        
        return img,label

    def _read_data(self,img,label):
        
        
            # supervised mode; if in supervised mode define a loader function 
            #that given the index of an image it returns the image and its 
            #categorical label
        img = self.transform(img)
        return img, label



def main():
    class dotdict(dict):
   
        __getattr__ = dict.get
        __setattr__ = dict.__setitem__
        __delattr__ = dict.__delitem__

    def load_yaml(config_file,config_type='dict'):
        with open(config_file) as f:
            cfg = yaml.safe_load(f)
        #params = yaml.load(f,Loader=yaml.FullLoader)
        
        if config_type=='object':
            cfg = dotdict(cfg)
        return cfg
    
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    torch.manual_seed(0)
    config_path = r'C:\Users\Safwen\.spyder-py3\BYOL\PyTorch-BYOL-master\config\config_sl.yaml'
    cfg = load_yaml(config_path,config_type='object') 
    
    config = yaml.load(open("./config/config.yaml", "r"), Loader=yaml.FullLoader)
    

    #device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #device='cpu'
    print(f"Training with: {device}")
    
    data_transform = get_simclr_data_transforms(**config['data_transforms'])
    transfor=TransformsSimCLR(size=96)
    normalize = transforms.Normalize(mean=[0.51290169, 0.51136089, 0.49742605], std=[0.21390466, 0.22544737, 0.24699091])

    transform_2 = transforms.Compose([
                transforms.RandomResizedCrop(96, scale=(0.08, 1.)),
                transforms.RandomHorizontalFlip(),
                transforms.Resize((cfg.img_sz,cfg.img_sz)),
                transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([GaussianBlur()], p=0.1),
                transforms.RandomApply([ImageOps.solarize], p=0.2),
                transforms.ToTensor(),
                normalize,
            ])
    
    annotation_file = 'data_recognition_train.csv'                                 
    
    train_dataset = FlowersDataset(cfg,annotation_file,\
                            data_type='train',transform=transfor)
    print('train data load success')
    
    train_loader = DataLoader(train_dataset,batch_size=cfg.batch_size,\
                            drop_last=True,num_workers=0)
        
    #resnet =  models.resnet18(pretrained=False)
    resnet = get_model(cfg)
    print('model loaded')
    model = BYOL(resnet, image_size=96, hidden_layer="avgpool")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

# solver
    global_step = 0
    for epoch in range(200):
        print("epoch number "+ str(epoch))
        metrics = defaultdict(list)
        for step, ((x_i, x_j), _) in enumerate(train_loader):
            
            x_i = x_i.to(device)
            x_j = x_j.to(device)

            loss = model(x_i, x_j)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.update_moving_average()  # update moving average of target encoder

            if step % 1 == 0:
                print(f"Step [{step}/{len(train_loader)}]:\tLoss: {loss.item()}")

            metrics["Loss/train"].append(loss.item())
            global_step += 1

    # write metrics to TensorBoard
        #print(f"Epoch [{epoch}/{args.num_epochs}]: " + "\t".join([f"{k}: {np.array(v).mean()}" for k, v in metrics.items()]))

        if epoch % 2 == 0:
            print(f"Saving model at epoch {epoch}")
            torch.save(resnet.state_dict(), f"experiments/self-supervised/model-pretrained-{epoch}.pth")


# save your improved network
    torch.save(resnet.state_dict(), "experiments/self-supervised/model_pretrained_final.pth")

if __name__ == '__main__':
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    main()
    