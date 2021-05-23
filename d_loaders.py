# -*- coding: utf-8 -*-
"""
Created on Sat May  8 13:27:47 2021

@author: Safwen
"""

'''
A script for loading the data and serving it to the model for pretraining
'''

import os,sys
sys.path.append(os.path.join(os.path.dirname("__file__"),'.'))
from data.transforms import get_simclr_data_transforms
#Apparently 512 is the maximum in python. I found the solution here- https://stackoverflow.com/a/28212496/8875017
from PIL import ImageFilter, ImageOps
import numpy as np
from torch.utils.data.dataloader import default_collate
import torchvision.transforms as transforms
from torch.utils.data import DataLoader 
from torch.utils.data.sampler import SubsetRandomSampler
import albumentations
from our_dataset import FlowersDataset,rotnet_collate_fn
#from utils.transformations import TransformsSimCLR
import utils
from utils.helpers import visualize
import yaml
from simclr_tran import TransformsSimCLR

config = yaml.load(open("./config/config.yaml", "r"), Loader=yaml.FullLoader)

class GaussianBlur(object):
    """Gaussian Blur version 2"""

    def __call__(self, x):
        sigma = np.random.uniform(0.1, 2.0)
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x
    
def get_dataloaders(cfg,val_split=None):
    
    train_dataloaders,val_dataloaders,test_dataloaders = loaders(cfg)

    return train_dataloaders,val_dataloaders,test_dataloaders

def get_datasets(cfg):
    
    train_dataset,val_dataset,test_dataset = loaders(cfg,get_dataset=True,)

    return train_dataset,test_dataset

def loaders(cfg,get_dataset=False):
    
    if cfg.data_aug:
        s=1
        color_jitter = transforms.ColorJitter(
            0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s
        )
        train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(size=96),
                transforms.RandomHorizontalFlip(),  # with 0.5 probability
                transforms.RandomApply([color_jitter], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
            ]
        )
        transfor=TransformsSimCLR(size=96)
        normalize = transforms.Normalize(mean=[0.51290169, 0.51136089, 0.49742605], std=[0.21390466, 0.22544737, 0.24699091])
        data_transform = get_simclr_data_transforms(**config['data_transforms'])
        transform_1 = transforms.Compose([
                transforms.RandomResizedCrop(96, scale=(0.08, 1.)),
                
                transforms.RandomHorizontalFlip(),
                transforms.Resize((cfg.img_sz,cfg.img_sz)),
                transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([GaussianBlur()], p=1.0),
                
                normalize,
            ])
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
            
        transform = (transform_1, transform_2)
        
        #train_transform = transforms.Compose([
    #transforms.RandomHorizontalFlip(p=0.5),
    #transforms.RandomVerticalFlip(p=0.5),
    #transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
    #transforms.RandomRotation(45)
    
#])
        #data_aug = albumentations.Compose([
        #        albumentations.Resize(224, 224, always_apply=True),
        #        albumentations.HorizontalFlip(p=1.0),
        #        albumentations.ShiftScaleRotate(
        #            shift_limit=0.3,
        #            scale_limit=0.3,
        #            rotate_limit=30,
        #            p=1.0
        #        ),
        #        albumentations.Normalize(mean=[0.485, 0.456, 0.406],
        #                  std=[0.229, 0.224, 0.225], always_apply=True)
        #    ])
                #train_transform = transforms.Compose([
    #transforms.RandomHorizontalFlip(p=0.5),
    #transforms.RandomCrop(32, padding=4),
    #transforms.CenterCrop(size=(10,20)),
    #transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
    #transforms.Grayscale(num_output_channels=1),
    #transforms.Pad(50, fill=0, padding_mode='constant'),
    #transforms.RandomVerticalFlip(p=1),
    #transforms.RandomPerspective(distortion_scale=0.5, p=1, interpolation=3, fill=0),
    #transforms.RandomRotation(degrees = 45),
    #transforms.RandomErasing(p=1)

        #data_aug = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),
        #transforms.RandomCrop(32, padding=4)])
    
        if cfg.mean_norm == True:
#            transform = transforms.Compose([transforms.Resize((cfg.img_sz,cfg.img_sz)),data_aug,
#                                        transforms.ToTensor(),
#                                        transforms.Normalize(mean=cfg.mean_pix, std=cfg.std_pix)])
            transform = transforms.Compose([data_aug,transforms.ToTensor(),
                                        transforms.Normalize(mean=cfg.mean_val, std=cfg.std_val)])       
#        transform = transforms.Compose([transforms.Resize((cfg.img_sz,cfg.img_sz)),data_aug,
#                                        transforms.ToTensor()])                 
        #transform = transforms.Compose([transform_2,transforms.ToTensor()])   
        #transform = data_aug
        #transform = transform
    elif cfg.mean_norm:
        transform = transforms.Compose([transforms.Resize((cfg.img_sz,cfg.img_sz)),
                                        transforms.ToTensor(),transforms.Normalize(mean=cfg.mean_pix, std=cfg.std_pix)])
    else:
        transform = transforms.Compose([transforms.Resize((cfg.img_sz,cfg.img_sz)),
                                        transforms.ToTensor()])   
       
    #transform_test = transforms.Compose([
    #transforms.ColorJitter(hue=.05, saturation=.05),
    #transforms.RandomHorizontalFlip(p=0.5),
    #transforms.RandomCrop(32, padding=4),
    #transforms.RandomVerticalFlip(p=0.5),
    #transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
    #transforms.CenterCrop(size=(10,20)),
    #transforms.RandomRotation(45),
    #transforms.ToTensor()
    
#])
    
    test_transform = transforms.Compose([
    transforms.Resize((cfg.img_sz,cfg.img_sz)),
    transforms.ToTensor()
])
    #transform_test = data_aug
    if cfg.pretext=='rotation':
        collate_func=rotnet_collate_fn
    else:
        collate_func=default_collate

    annotation_file = 'small_labeled_data.csv'                                 
    
    train_dataset = FlowersDataset(cfg,annotation_file,\
                            data_type='train',transform=transform_2)
    
    
    val_dataset=None
    
    
    annotation_file = 'data_recognition_test.csv'                                  
    
    test_dataset = FlowersDataset(cfg,annotation_file,\
                                  data_type='test',transform=test_transform)
        
    #val_dataset=test_dataset	
    # if you want to use a portion of training dataset as validation data
    if cfg.val_split:
        
#        shuffle_dataset = True
        random_seed= 42

        # Creating data indices for training and validation splits:
        dataset_size = len(train_dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(cfg.val_split * dataset_size))
        # shuffle dataset
        np.random.seed(random_seed)
        np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]

        # Creating PT data samplers and loaders:
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)

        #dataloader_train = DataLoader(train_dataset, batch_size=cfg.batch_size, 
        #                          collate_fn=collate_func,sampler=train_sampler)
        dataloader_train = DataLoader(train_dataset,batch_size=cfg.batch_size,\
                            collate_fn=collate_func,sampler=train_sampler)
        #dataloader_train = DataLoader(train_dataset,batch_size=cfg.batch_size,\
        #                    drop_last=False,num_workers=0, collate_fn=default_collate,sampler=train_sampler)
        
        #dataloader_val = DataLoader(train_dataset, batch_size=cfg.batch_size,
        #                               collate_fn=collate_func,sampler=valid_sampler)
        dataloader_val = DataLoader(train_dataset,batch_size=cfg.batch_size,\
                            collate_fn=collate_func,sampler=valid_sampler)
        #dataloader_val = DataLoader(train_dataset,batch_size=cfg.batch_size,\
        #                    drop_last=False,num_workers=0, collate_fn=default_collate,sampler=valid_sampler)
    
    else:
        #dataloader_train = DataLoader(train_dataset,batch_size=cfg.batch_size,\
        #                    collate_fn=collate_func,shuffle=True)
        dataloader_train = DataLoader(train_dataset,batch_size=cfg.batch_size,\
                            collate_fn=collate_func,shuffle=True)
        if val_dataset:
            # if you have separate val data define val loader here
            #dataloader_val = DataLoader(val_dataset,batch_size=cfg.batch_size,\
            #                    collate_fn=collate_func,shuffle=True)
            dataloader_val = DataLoader(train_dataset,batch_size=cfg.batch_size,\
                            collate_fn=collate_func,shuffle=True)
        else:
            dataloader_val=None
    
    if get_dataset:
        
        return train_dataset,val_dataset,test_dataset
        
    dataloader_test = DataLoader(test_dataset,batch_size=cfg.batch_size,\
                            collate_fn=collate_func,shuffle=True)
    
    return dataloader_train,dataloader_val, dataloader_test
#%%
import yaml
if __name__=='__main__':
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
    
    config = yaml.load(open("./config/config.yaml", "r"), Loader=yaml.FullLoader)
    config_file=r'C:\Users\Safwen\.spyder-py3\BYOL\PyTorch-BYOL-master\config\config_sl.yaml'
    cfg = load_yaml(config_file,config_type='object')
    
#    tr_dset,ts_dset = get_datasets(cfg)

    tr_loaders,val_loaders,ts_loaders = get_dataloaders(cfg)
        
    #print ('length of tr_dset: {}'.format(len(tr_dset)))
    #print ('length of ts_dset: {}'.format(len(ts_dset)))

    
    data, label = next(iter(tr_loaders))
    print(data.shape, label) 

    data, label = next(iter(ts_loaders))
    print(data.shape, label)    
    
    visualize(data.numpy(),label) 
